from pathlib import Path
from typing import List

import jax
import numpy as np
import tensorflow as tf
import typer
from flax import nnx
from jax import numpy as jnp
from keras_cv import layers
from keras_cv.src import bounding_box
from keras_cv.src.metrics.coco import compute_pycoco_metrics

from jolo.data.dataset import get_dataset
from jolo.nn import utils
from jolo.nn.losses import yolo_loss
from jolo.nn.model import Yolo
from jolo.nn.optimizer import build_optimizer
from jolo.utils.box_utils import vec2box


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


@nnx.jit
def train_step(
    model: Yolo,
    optimizer: nnx.Optimizer,
    scalers: jax.Array,
    anchors: jax.Array,
    anchors_norm: jax.Array,
    batch,
):
    loss, grads = nnx.value_and_grad(yolo_loss)(
        model, batch, anchors_norm, anchors, scalers
    )
    optimizer.update(grads)
    return loss


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
):
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
    for epoch in range(epochs):
        step_losses = []
        model.train()
        for x_tf, y_tf in train_dataset:
            y = {}
            x = jnp.asarray(x_tf.numpy())
            for k, v in y_tf.items():
                y[k] = jnp.asarray(v.numpy())
            step_loss = train_step(
                model, optimizer, scalers, anchors, anchors_norm, (x, y)
            )
            step_losses.append(step_loss / batch_size)
        print(f"{epoch+1} - loss: {np.mean(step_loss)}")
        model.eval()

        val_pclasses = []
        val_pboxes = []
        val_yclasses = []
        val_yboxes = []
        for val_xtf, val_ytf in val_dataset:
            val_y = {}
            val_x = jnp.asarray(val_xtf.numpy())
            for k, v in val_ytf.items():
                val_y[k] = jnp.asarray(v.numpy())
            val_class, _, val_vector = model(val_x)
            val_vector_concatenated = jax.numpy.concatenate(
                [
                    jax.numpy.reshape(
                        v, (v.shape[0], v.shape[1] * v.shape[2], v.shape[3])
                    )
                    for v in val_vector
                ],
                axis=1,
            )
            val_class_concatenated = jax.numpy.concatenate(
                [
                    jax.numpy.reshape(
                        cls, (cls.shape[0], cls.shape[1] * cls.shape[2], cls.shape[3])
                    )
                    for cls in val_class
                ],
                axis=1,
            )
            val_pbox = (
                vec2box(val_vector_concatenated, anchors_norm) * scalers[..., None]
            )
            val_pclasses.append(val_class_concatenated)
            val_pboxes.append(val_pbox)
            val_yclasses.append(val_y["classes"])
            val_yboxes.append(val_y["bboxes"])

        _ = evaluate_coco_metrics(
            val_yclasses,
            val_yboxes,
            val_pclasses,
            val_pboxes,
            nms_conf,
            nms_iou,
            nms_max_detections,
        )


if __name__ == "__main__":
    typer.run(main)
