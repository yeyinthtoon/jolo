import math
from typing import List, Literal, Tuple, Union

import jax
from flax import typing as flax_typing


def calculate_iou_jax(
    bbox1: jax.Array,
    bbox2: jax.Array,
    dtype: flax_typing.Dtype,
    metrics: Literal["iou", "diou", "ciou", "siou"] = "iou",
    pairwise: bool = True,
    eps: float = 1e-7,
) -> jax.Array:
    """
    Calculates IoU (Intersection over Union), DIoU, CIoU, SIoU between bounding boxes.

    Args:
        bbox1: First set of bounding boxes, shape [..., 4] or [..., A, 4].
        bbox2: Second set of bounding boxes, shape [..., 4] or [..., B, 4].
        metrics: The metric to calculate ("iou", "diou", "ciou", "siou"). Defaults to "iou".

    Returns:
        Tensor containing the calculated IoU/DIoU/CIoU.
    """

    bbox1 = bbox1.astype(dtype=dtype)
    bbox2 = bbox2.astype(dtype=dtype)

    if pairwise:
        # Expand dimensions if necessary for broadcasting
        if len(bbox1.shape) == 2 and len(bbox2.shape) == 2:
            bbox1 = bbox1[:, None, :]
            bbox2 = bbox2[None, :, :]
        elif len(bbox1.shape) == 3 and len(bbox2.shape) == 3:
            bbox1 = bbox1[:, :, None, :]  # (B, A, 4) -> (B, A, 1, 4)
            bbox2 = bbox2[:, None, :, :]  # (B, B, 4) -> (B, 1, B, 4)

    # Calculate intersection coordinates
    xmin_inter = jax.numpy.maximum(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = jax.numpy.maximum(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = jax.numpy.minimum(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = jax.numpy.minimum(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    x_min = 0.0
    intersection_area = jax.numpy.maximum(
        xmax_inter - xmin_inter, x_min
    ) * jax.numpy.maximum(ymax_inter - ymin_inter, x_min)

    # Calculate area of each bbox
    w1 = bbox1[..., 2] - bbox1[..., 0]
    h1 = bbox1[..., 3] - bbox1[..., 1]
    w2 = bbox2[..., 2] - bbox2[..., 0]
    h2 = bbox2[..., 3] - bbox2[..., 1]
    area_bbox1 = w1 * h1
    area_bbox2 = w2 * h2

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + eps)

    if metrics == "iou":
        return iou

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = jax.numpy.maximum(bbox1[..., 2], bbox2[..., 2]) - jax.numpy.minimum(
        bbox1[..., 0], bbox2[..., 0]
    )
    c_y = jax.numpy.maximum(bbox1[..., 3], bbox2[..., 3]) - jax.numpy.minimum(
        bbox1[..., 1], bbox2[..., 1]
    )
    if metrics in ["diou", "ciou"]:
        diag_dis = c_x**2 + c_y**2 + eps
        diou = iou - (cent_dis / diag_dis)
        if metrics == "diou":
            return diou

        # Compute aspect ratio penalty term
        arctan = jax.numpy.arctan(
            (bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + eps)
        ) - jax.numpy.arctan(
            (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + eps)
        )
        v = (4 / (math.pi**2)) * (arctan**2)
        alpha = jax.lax.stop_gradient(v / (v - iou + 1 + eps))

        # Compute CIoU
        ciou = diou - alpha * v
        return ciou
    if metrics == "siou":
        sigma = cent_dis**0.5 + eps
        sin_alpha_1 = jax.numpy.abs(cx2 - cx1) / sigma
        sin_alpha_2 = jax.numpy.abs(cy2 - cy1) / sigma
        threshold = (2**0.5) / 2
        sin_alpha = jax.numpy.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = (
            1 - 2 * jax.numpy.sin(jax.numpy.arcsin(sin_alpha) - math.pi / 4) ** 2
        )

        rho_x = ((cx2 - cx1) / (c_x + eps)) ** 2
        rho_y = ((cy2 - cy1) / (c_y + eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - jax.numpy.exp(gamma * rho_x) - jax.numpy.exp(gamma * rho_y)

        omiga_w = jax.numpy.abs(w1 - w2) / jax.numpy.maximum(w1, w2)
        omiga_h = jax.numpy.abs(h1 - h2) / jax.numpy.maximum(h1, h2)
        shape_cost = jax.numpy.power(
            1 - jax.numpy.exp(-1 * omiga_w), 4
        ) + jax.numpy.power(1 - jax.numpy.exp(-1 * omiga_h), 4)
        return iou - 0.5 * (distance_cost + shape_cost)
    raise ValueError(f"Metric type {metrics} not supported.")


def get_valid_matrix_jax(anchors: jax.Array, target_bbox: jax.Array) -> jax.Array:
    """
    Calculate target on anchors matrix

    Args:
        anchors: anchors, shape [B, A, 4].
        target_bbox: ground truth boxes, shape [B, T, 4].
    Returns:
        Bool Tensor, shape [B, T, A].
    """
    xmin, ymin, xmax, ymax = jax.numpy.split(target_bbox, [1, 2, 3], axis=-1)
    anchors = anchors[None, None]
    anchors_x, anchors_y = jax.numpy.split(anchors, [1], axis=-1)

    anchors_x, anchors_y = (
        jax.numpy.squeeze(anchors_x, -1),
        jax.numpy.squeeze(anchors_y, -1),
    )

    target_in_x = (xmin < anchors_x) & (anchors_x < xmax)
    target_in_y = (ymin < anchors_y) & (anchors_y < ymax)

    target_on_anchor = target_in_x & target_in_y
    return target_on_anchor


def get_cls_matrix_jax(
    target_cls: jax.Array,
    predict_cls: jax.Array,
    gt_mask: jax.Array,
    from_logits: bool = True,
) -> jax.Array:
    """
    get target class score of  all anchors

    Args:
        predict_cls: class prediction, shape [B, A, C].
        target_cls: ground truth boxes, shape [B, T, C].
    Returns:
        Tensor, shape [B, T, A].
    """
    if from_logits:
        predict_cls = jax.nn.sigmoid(predict_cls)
    predict_cls = jax.numpy.transpose(predict_cls, (0, 2, 1))
    target_cls = jax.numpy.repeat(target_cls, predict_cls.shape[2], 2)
    gt_mask = jax.numpy.repeat(gt_mask, predict_cls.shape[2], 2)
    cls_probabilities = (
        jax.numpy.take_along_axis(predict_cls, target_cls, axis=1) * gt_mask
    )
    return cls_probabilities


def get_metrics_jax(
    target_cls: jax.Array,
    target_bbox: jax.Array,
    predict_cls: jax.Array,
    predict_bbox: jax.Array,
    target_anchor_mask: jax.Array,
    gt_mask: jax.Array,
    dtype: flax_typing.Dtype,
    iou: Literal["iou", "diou", "ciou", "siou"],
    iou_factor: float = 6,
    cls_factor: float = 0.5,
    from_logits: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    iou_matrix = jax.numpy.clip(
        calculate_iou_jax(target_bbox, predict_bbox, dtype, iou), 0.0, 1.0
    )
    cls_matrix = get_cls_matrix_jax(target_cls, predict_cls, gt_mask, from_logits)

    iou_matrix = iou_matrix * target_anchor_mask
    cls_matrix = cls_matrix * target_anchor_mask

    target_matrix = (iou_matrix**iou_factor) * (cls_matrix**cls_factor)
    return target_matrix, iou_matrix


def gather_topk_jax(
    target_matrix: jax.Array, topk=10, topk_mask=None, eps=1e-7
) -> jax.Array:
    topk_metrics, topk_idxs = jax.lax.top_k(target_matrix, topk)
    num_of_anchors = target_matrix.shape[-1]
    if topk_mask is None:
        topk_mask = jax.numpy.tile(
            (jax.numpy.max(topk_metrics, axis=-1, keepdims=True) > eps), [1, 1, topk]
        )
    topk_idxs = jax.numpy.where(topk_mask, topk_idxs, jax.numpy.zeros_like(topk_idxs))
    is_in_topk = jax.numpy.sum(jax.nn.one_hot(topk_idxs, num_of_anchors), axis=-2)
    is_in_topk = jax.numpy.where(
        is_in_topk > 1, jax.numpy.zeros_like(is_in_topk), is_in_topk
    )
    is_in_topk = is_in_topk.astype(target_matrix.dtype)
    return is_in_topk


def get_align_indices_and_valid_mask_jax(topk_mask, iou_matrix):
    valid_mask = jax.numpy.sum(topk_mask, -2)
    shapes = topk_mask.shape
    max_target = shapes[1]
    num_anchors = shapes[2]
    batch_size = shapes[0]
    condition = jax.numpy.max(valid_mask) > 1
    multi_assigned = jax.numpy.broadcast_to(
        valid_mask[:, None, :] > 1, (batch_size, max_target, num_anchors)
    )

    best_match_idx = jax.numpy.argmax(iou_matrix, axis=1)
    best_matches = jax.nn.one_hot(best_match_idx, max_target, axis=1)

    topk_mask = jax.numpy.where(
        condition, jax.numpy.where(multi_assigned, best_matches, topk_mask), topk_mask
    )
    valid_mask = jax.numpy.where(
        condition, jax.numpy.sum(topk_mask, axis=-2), valid_mask
    )
    aligned_indices = jax.numpy.argmax(topk_mask, axis=-2)
    return aligned_indices[..., None], valid_mask, topk_mask


def get_aligned_targets_detection_jax(
    target_cls,
    target_bbox,
    predict_cls,
    predict_bbox,
    number_of_classes,
    anchors,
    dtype,
    iou: Literal["iou", "diou", "ciou", "siou"],
    iou_factor: float = 6,
    cls_factor: float = 0.5,
    topk: int = 10,
    from_logits: bool = True,
):
    target_anchor_mask = get_valid_matrix_jax(anchors, target_bbox).astype(dtype)
    gt_mask = jax.numpy.sum(target_bbox, axis=-1) > 0
    gt_mask = gt_mask[:, :, None].astype(dtype)
    # print("gt_mask: ",gt_mask.shape)
    # print("target_anchor_mask: ", target_anchor_mask.shape)
    target_matrix, iou_matrix = get_metrics_jax(
        target_cls=target_cls,
        target_bbox=target_bbox,
        predict_cls=predict_cls,
        predict_bbox=predict_bbox,
        target_anchor_mask=target_anchor_mask * gt_mask,
        gt_mask=gt_mask,
        dtype=dtype,
        iou=iou,
        iou_factor=iou_factor,
        cls_factor=cls_factor,
        from_logits=from_logits,
    )
    topk_mask = gather_topk_jax(target_matrix, topk, gt_mask).astype(dtype)
    topk_mask = topk_mask * target_anchor_mask * gt_mask

    aligned_indices, valid_mask, topk_mask = get_align_indices_and_valid_mask_jax(
        topk_mask, iou_matrix
    )

    target_matrix = target_matrix * topk_mask

    align_bbox = jax.numpy.take_along_axis(
        target_bbox, jax.numpy.repeat(aligned_indices, 4, 2), axis=1
    )
    align_cls = jax.numpy.squeeze(
        jax.numpy.take_along_axis(target_cls, aligned_indices, axis=1), -1
    )
    align_cls = jax.nn.one_hot(align_cls, number_of_classes, dtype=dtype)

    max_target = jax.numpy.max(target_matrix, axis=-1, keepdims=True)
    max_iou = jax.numpy.max(iou_matrix * topk_mask, axis=-1, keepdims=True)
    normalize_term = (target_matrix / (max_target + 1e-7)) * max_iou
    normalize_term = jax.numpy.transpose(normalize_term, (0, 2, 1))
    normalize_term = jax.numpy.take_along_axis(normalize_term, aligned_indices, axis=2)

    align_cls = (
        align_cls * normalize_term * valid_mask.astype(normalize_term.dtype)[:, :, None]
    )

    return (
        jax.lax.stop_gradient(align_cls),
        jax.lax.stop_gradient(align_bbox),
        jax.lax.stop_gradient(valid_mask),
        jax.lax.stop_gradient(aligned_indices),
    )


def xywh2xyxy(bbox: List[Union[float, int]]) -> List[Union[float, int]]:
    """Convert bbox from xywh format to xyxy format"""
    x, y, w, h = bbox
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return [x1, y1, x2, y2]


def get_normalized_box_area_jax(boxes, image_h, image_w):
    scale = jax.numpy.asarray([image_w, image_h, image_w, image_h])
    boxes = boxes / scale
    w1 = boxes[..., 2] - boxes[..., 0]
    h1 = boxes[..., 3] - boxes[..., 1]
    area = w1 * h1
    return area


def generate_bbox_mask_jax(bboxes, mask_h, mask_w, image_h, image_w, dtype):
    scale = jax.numpy.asarray(
        [mask_w / image_w, mask_h / image_h, mask_w / image_w, mask_h / image_h]
    )

    bboxes = bboxes * scale
    y, x = jax.numpy.meshgrid(
        jax.numpy.arange(mask_h, dtype=dtype),
        jax.numpy.arange(mask_w, dtype=dtype),
        indexing="ij",
    )
    max_detect = bboxes.shape[1]
    batch = bboxes.shape[0]
    x = x[None, :, :, None]
    y = y[None, :, :, None]

    x = jax.numpy.broadcast_to(x, [batch, mask_h, mask_w, max_detect])
    y = jax.numpy.broadcast_to(y, [batch, mask_h, mask_w, max_detect])

    xmin, ymin, xmax, ymax = jax.numpy.unstack(bboxes, axis=-1)

    # Reshape for broadcasting
    xmin = xmin[:, None, None, :]
    ymin = ymin[:, None, None, :]
    xmax = xmax[:, None, None, :]
    ymax = ymax[:, None, None, :]
    mask = jax.numpy.logical_and(
        jax.numpy.logical_and(x >= xmin, x < xmax),
        jax.numpy.logical_and(y >= ymin, y < ymax),
    )
    mask = mask.astype(dtype)
    return mask


def vec2box(vecs, anchors_norm):
    lt, rb = jax.numpy.split(vecs, 2, axis=-1)
    preds_box = jax.numpy.concatenate([anchors_norm - lt, anchors_norm + rb], axis=-1)
    return preds_box
