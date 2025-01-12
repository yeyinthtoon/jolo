import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from functools import partial
from pathlib import Path
from typing import List

import jax
import numpy as np
import tensorflow as tf
import typer
from flax import nnx
from jax import numpy as jnp
from jax.experimental import jax2tf, mesh_utils
from keras_cv import layers
from keras_cv.src import bounding_box
from keras_cv.src.metrics.coco import compute_pycoco_metrics

from jolo.data.dataset import get_dataset
from jolo.nn import utils
from jolo.nn.losses import forward_with_loss
from jolo.nn.model import Yolo
from jolo.nn.optimizer import build_optimizer


def decode_prediction(classes, boxes, nms_conf, nms_iou, nms_max_detections):
    classes_tf = tf.convert_to_tensor(jax.numpy.concat(classes, axis=0))
    boxes_tf = tf.convert_to_tensor(jax.numpy.concat(boxes, axis=0))

    nms = layers.NonMaxSuppression(
        "xyxy",
        True,
        confidence_threshold=nms_conf,
        iou_threshold=nms_iou,
        max_detections=nms_max_detections,
        dtype="float32",
    )
    val_predictions = nms(boxes_tf, classes_tf)

    pred_boxes = val_predictions["boxes"].numpy()
    pred_classes = val_predictions["classes"].numpy()
    pred_confidence = val_predictions["confidence"].numpy()
    pred_num_detections = val_predictions["num_detections"].numpy()
    return pred_boxes, pred_classes, pred_confidence, pred_num_detections


def evaluate_coco_metrics(
    classes, boxes, pclasses, pboxes, nms_conf, nms_iou, nms_max_detections
):
    classes_np = np.squeeze(tf.concat(classes, axis=0).numpy())
    boxes = tf.concat(boxes, axis=0)
    pred_boxes, pred_classes, pred_confidence, pred_num_detections = decode_prediction(
        pclasses, pboxes, nms_conf, nms_iou, nms_max_detections
    )
    boxes_np = bounding_box.convert_format(boxes, source="xyxy", target="yxyx").numpy()

    pred_boxes = bounding_box.convert_format(
        pred_boxes, source="xyxy", target="yxyx"
    ).numpy()

    source_ids = np.char.mod(
        "%d", np.linspace(1, len(pred_confidence), len(pred_confidence))
    )
    valid_gts = (boxes_np.sum(-1) > 0).sum(-1)
    ground_truth = {
        "source_id": [source_ids],
        "height": [np.tile(np.array([640]), len(valid_gts))],
        "width": [np.tile(np.array([640]), len(valid_gts))],
        "num_detections": [valid_gts],
        "boxes": [boxes_np],
        "classes": [classes_np],
    }

    predictions = {
        "source_id": [source_ids],
        "detection_boxes": [pred_boxes],
        "detection_classes": [pred_classes],
        "detection_scores": [pred_confidence],
        "num_detections": [pred_num_detections],
    }
    metrics = compute_pycoco_metrics(ground_truth, predictions)
    return metrics


def evaluate_coco_metrics_no_nms(
    classes, boxes, pred_classes, pred_boxes, pred_confidence, pred_num_detections
):
    classes = jax.numpy.concat(classes, axis=0).squeeze(-1)
    boxes = jax.numpy.concat(boxes, axis=0)

    pred_classes = jax.numpy.concat(pred_classes, axis=0)
    pred_confidence = jax.numpy.concat(pred_confidence, axis=0)
    pred_num_detections = jax.numpy.concat(pred_num_detections, axis=0)
    pred_boxes = jax.numpy.concat(pred_boxes, axis=0)

    boxes_np = bounding_box.convert_format(boxes, source="xyxy", target="yxyx").numpy()
    pred_boxes = bounding_box.convert_format(
        pred_boxes, source="xyxy", target="yxyx"
    ).numpy()

    source_ids = np.char.mod(
        "%d", np.linspace(1, len(pred_confidence), len(pred_confidence))
    )
    valid_gts = (boxes_np.sum(-1) > 0).sum(-1)
    ground_truth = {
        "source_id": [source_ids],
        "height": [np.tile(np.array([640]), len(valid_gts))],
        "width": [np.tile(np.array([640]), len(valid_gts))],
        "num_detections": [valid_gts],
        "boxes": [boxes_np],
        "classes": [np.array(classes)],
    }

    predictions = {
        "source_id": [source_ids],
        "detection_boxes": [np.array(pred_boxes)],
        "detection_classes": [np.array(pred_classes)],
        "detection_scores": [np.array(pred_confidence)],
        "num_detections": [np.array(pred_num_detections)],
    }
    metrics = compute_pycoco_metrics(ground_truth, predictions)
    return metrics


@nnx.jit
def train_step(
    model: Yolo,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    scalers: jax.Array,
    anchors: jax.Array,
    anchors_norm: jax.Array,
    batch,
):
    (_, outputs), grads = nnx.value_and_grad(forward_with_loss, has_aux=True)(
        model, batch, anchors_norm, anchors, scalers
    )
    metrics.update(**outputs["losses"])
    optimizer.update(grads)


@nnx.jit
def validation_step(
    model: Yolo,
    metrics: nnx.MultiMetric,
    scalers: jax.Array,
    anchors: jax.Array,
    anchors_norm: jax.Array,
    batch,
):
    _, outputs = forward_with_loss(model, batch, anchors_norm, anchors, scalers)
    metrics.update(**outputs["losses"])
    return outputs["predictions"]


def main(
    train_data_dir: Path,
    val_data_dir: Path,
    num_classes: int,
    batch_size: int,
    steps_per_epoch: int,
    head_bias_init: List[float] = typer.Option(..., "--bias-init"),
    gradient_accumulation_step: int = 1,
    dtype: str = "float32",
    random_seed: int = 0,
    input_image_size: int = 640,
    nms_conf: float = 1e-3,
    nms_iou: float = 0.6,
    nms_max_detections: int = 100,
    epochs: int = 100,
    tpu: bool = False,
):
    nms_fn = partial(tf.image.combined_non_max_suppression, clip_boxes=False)
    train_tfrecs = list(map(str, train_data_dir.glob("*.tfrecord")))
    train_dataset = get_dataset(train_tfrecs, batch_size, training=True)

    val_tfrecs = list(map(str, val_data_dir.glob("*.tfrecord")))
    val_dataset = get_dataset(val_tfrecs, batch_size, training=False)

    model = Yolo(
        num_classes=num_classes,
        head_bias_init=head_bias_init,
        dtype=dtype,
        rngs=nnx.Rngs(random_seed),
    )
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        box_loss=nnx.metrics.Average("box_loss"),
        class_loss=nnx.metrics.Average("class_loss"),
        dfl_loss=nnx.metrics.Average("dfl_loss"),
    )
    data = np.random.randn(0, input_image_size, input_image_size, 3).astype("float32")
    model.eval()
    classes_shape, _, _ = jax.eval_shape(model, data)
    anchors, scalers = utils.get_anchors_and_scalers(
        classes_shape, (input_image_size, input_image_size)
    )
    anchors_norm = anchors / scalers[..., None]
    model.train()
    optimizer = build_optimizer(
        model,
        steps_per_epoch=steps_per_epoch,
        gradient_accumulation_step=gradient_accumulation_step,
        epochs=epochs,
    )
    if tpu:
        num_devices = jax.local_device_count()
        mesh = jax.sharding.Mesh(
            mesh_utils.create_device_mesh((num_devices,)), ("data",)
        )
        model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        state = nnx.state((model, optimizer, metrics))
        state = jax.device_put(state, model_sharding)
        nnx.update((model, optimizer, metrics), state)
        anchors, anchors_norm, scalers = jax.device_put(
            (anchors, anchors_norm, scalers), model_sharding
        )

    for epoch in range(epochs):
        model.train()
        for x_tf, y_tf in train_dataset:
            y = {}
            x = jnp.asarray(x_tf.numpy())
            for k, v in y_tf.items():
                y[k] = jnp.asarray(v.numpy())
            if tpu:
                x, y = jax.device_put((x, y), data_sharding)
            train_step(
                model, optimizer, metrics, scalers, anchors, anchors_norm, (x, y)
            )

        print(f"[train] epoch: {epoch+1},", end=" ")
        for metric, value in metrics.compute().items():
            print(f"{metric} : {value}", end=" ")
        print()
        metrics.reset()
        model.eval()

        val_pclasses = []
        val_pboxes = []
        val_pscores = []
        val_pnd = []
        val_yclasses = []
        val_yboxes = []
        for val_xtf, val_ytf in val_dataset:
            val_y = {}
            val_x = jnp.asarray(val_xtf.numpy())
            for k, v in val_ytf.items():
                val_y[k] = jnp.asarray(v.numpy())
            if tpu:
                val_x, val_y = jax.device_put((val_x, val_y), data_sharding)
            val_predictions = validation_step(
                model, metrics, scalers, anchors, anchors_norm, (val_x, val_y)
            )
            val_nmsed = jax2tf.call_tf(nms_fn)(
                val_predictions["boxes"][:, :, None, :].astype("float32"),
                jax.nn.sigmoid(val_predictions["classes"]).astype("float32"),
                nms_max_detections,
                nms_max_detections,
                nms_iou,
                nms_conf,
            )
            val_pclasses.append(val_nmsed.nmsed_classes)
            val_pboxes.append(val_nmsed.nmsed_boxes)
            val_pscores.append(val_nmsed.nmsed_scores)
            val_pnd.append(val_nmsed.valid_detections)

            val_yclasses.append(val_y["classes"])
            val_yboxes.append(val_y["bboxes"])

        print(f"[validation] epoch: {epoch+1},", end=" ")
        for metric, value in metrics.compute().items():
            print(f"{metric} : {value}", end=" ")
        print()
        metrics.reset()

        _ = evaluate_coco_metrics_no_nms(
            val_yclasses, val_yboxes, val_pclasses, val_pboxes, val_pscores, val_pnd
        )


if __name__ == "__main__":
    typer.run(main)
