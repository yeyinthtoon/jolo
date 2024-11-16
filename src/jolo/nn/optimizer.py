from collections.abc import Mapping

import optax
from flax import nnx

from jolo.nn.model import Yolo


def get_param_labels(nested_dict, parent=None):
    param_labels = {}
    for k, v in nested_dict.items():
        if isinstance(v, Mapping):
            param_labels[k] = get_param_labels(v, parent=k)
        elif k == "bias":
            param_labels[k] = "bias"
        elif "norm" in parent:
            param_labels[k] = "norm"
        else:
            param_labels[k] = "params"
    return param_labels


def build_optimizer(
    model: Yolo,
    steps_per_epoch,
    gradient_accumulation_step: int = 1,
    init_lr: float = 1e-3,
    min_lr_multiplier: float = 0.01,
    epochs: int = 100,
    weight_decay: float = 0.0005,
):
    _, params, *_ = nnx.split(model, nnx.Param, nnx.BatchStat)
    param_labels = get_param_labels(params)
    lr_schedule = optax.schedules.cosine_decay_schedule(
        init_lr, int(steps_per_epoch * epochs), alpha=min_lr_multiplier
    )
    optimizer = optax.multi_transform(
        {
            "bias": optax.adam(lr_schedule),
            "norm": optax.adam(lr_schedule),
            "params": optax.adamw(lr_schedule, weight_decay=weight_decay),
        },
        nnx.State(param_labels),
    )
    if gradient_accumulation_step > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=gradient_accumulation_step
        )
    optimizer = nnx.Optimizer(model, optimizer)
    return optimizer
