from typing import Tuple

import jax
import optax

from jolo.nn.model import Yolo
from jolo.utils.box_utils import (
    calculate_iou_jax,
    get_aligned_targets_detection_jax,
    vec2box,
)


def forward_with_loss(
    model: Yolo,
    batch: Tuple[jax.Array, jax.Array],
    anchors_norm: jax.Array,
    anchors: jax.Array,
    scalers: jax.Array,
):
    x, y = batch
    classes, anchors_out, vector = model(x)
    vector_concat = jax.numpy.concatenate(
        [
            jax.numpy.reshape(v, (v.shape[0], v.shape[1] * v.shape[2], v.shape[3]))
            for v in vector
        ],
        axis=1,
    )
    classes_concat = jax.numpy.concatenate(
        [
            jax.numpy.reshape(
                cls, (cls.shape[0], cls.shape[1] * cls.shape[2], cls.shape[3])
            )
            for cls in classes
        ],
        axis=1,
    )
    anchors_concat = jax.numpy.concatenate(
        [
            jax.numpy.reshape(
                anchor,
                (anchor.shape[0], anchor.shape[1] * anchor.shape[2], anchor.shape[3]),
            )
            for anchor in anchors_out
        ],
        axis=1,
    )
    boxes = vec2box(vector_concat, anchors_norm)
    align_cls, align_bbox, valid_mask, _ = get_aligned_targets_detection_jax(
        target_cls=y["classes"],
        target_bbox=y["bboxes"],
        predict_cls=classes_concat,
        predict_bbox=boxes * scalers[..., None],
        number_of_classes=4,
        anchors=anchors,
        iou="ciou",
        dtype="float32",
        from_logits=True,
    )

    align_bbox = align_bbox / scalers[..., None]
    cls_norm = jax.numpy.maximum(jax.numpy.sum(align_cls), 1.0)
    box_norm = jax.numpy.sum(align_cls, axis=-1) * valid_mask

    class_loss = 0.5 * jax.numpy.sum(
        (sigmoid_loss_fn(align_cls, classes_concat) / cls_norm)
    )
    # TODO: use where on sum
    """
        jax.numpy.sum(
        (
            box_loss_fn(
                align_bbox,
                boxes,
                "float32",
            )
            * box_norm,
            where=valid_mask
        )
    """
    box_loss = 7.5 * jax.numpy.sum(
        (
            box_loss_fn(
                align_bbox,
                boxes,
                valid_mask,
                "float32",
            )
            * box_norm
        )
        / cls_norm
    )
    dfl_loss = 1.5 * jax.numpy.sum(
        (
            dfl_loss_fn(align_bbox, anchors_concat, valid_mask, anchors_norm, 16)
            * box_norm
        )
        / cls_norm
    )

    loss = class_loss + box_loss + dfl_loss
    return loss * x.shape[0], {
        "predictions": {"classes": classes_concat, "boxes": boxes * scalers[..., None]},
        "losses": {
            "loss": loss,
            "box_loss": box_loss,
            "class_loss": class_loss,
            "dfl_loss": dfl_loss,
        },
    }


def box_loss_fn(y_true, y_pred, valid_mask, dtype):
    y_true = y_true * valid_mask[..., None]
    y_pred = y_pred * valid_mask[..., None]
    iou = calculate_iou_jax(y_true, y_pred, dtype, metrics="ciou", pairwise=False)
    iou_loss = 1.0 - iou
    return iou_loss


def sigmoid_loss_fn(y_true, y_pred):
    return jax.numpy.sum(
        optax.losses.sigmoid_binary_cross_entropy(y_pred, y_true), axis=(1, 2)
    )


def dfl_loss_fn(y_true, y_pred, valid_mask, anchor_norm, reg_max):
    b, total, channel = y_pred.shape
    y_pred = jax.lax.reshape(y_pred, (b, total, 4, channel // 4))
    left_target, right_target = jax.numpy.split(y_true, 2, axis=-1)
    target_dist = jax.numpy.concatenate(
        [(anchor_norm - left_target), (right_target - anchor_norm)], axis=-1
    )
    target_dist = jax.numpy.clip(target_dist, 0.0, reg_max - 1.01)
    target_left, target_right = (
        jax.numpy.floor(target_dist),
        jax.numpy.floor(target_dist) + 1,
    )
    weight_left, weight_right = (
        target_right - target_dist,
        target_dist - target_left,
    )
    target_left = jax.nn.one_hot(target_left, reg_max)
    target_right = jax.nn.one_hot(target_right, reg_max)
    loss_left = (
        optax.losses.safe_softmax_cross_entropy(y_pred, target_left)
        * valid_mask[..., None]
    )
    loss_right = (
        optax.losses.safe_softmax_cross_entropy(y_pred, target_right)
        * valid_mask[..., None]
    )
    dfl_loss = loss_left * weight_left + loss_right * weight_right
    # TODO: Use where at mean
    dfl_loss = jax.numpy.mean(dfl_loss, axis=-1)
    return dfl_loss
