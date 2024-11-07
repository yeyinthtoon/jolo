from typing import Literal, Tuple, TypeAlias, Union, List

import jax
from jax import ShapeDtypeStruct
from flax import nnx

SingleIntOrDubleIntTuple: TypeAlias = Union[int, Tuple[int, int]]
DoubleIntTuple: TypeAlias = Tuple[int, int]
PaddingLike: TypeAlias = Union[str, Tuple[int, int]]
Activation: TypeAlias = Literal[
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "one_hot",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "standardize",
    "swish",
    "tanh",
]


def auto_pad(
    kernel_size: DoubleIntTuple, dilation: SingleIntOrDubleIntTuple = 1
) -> Tuple[int, int]:
    """
    Auto Padding for the convolution blocks
    """
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def get_activation(activation: str):
    try:
        return getattr(nnx, activation)
    except AttributeError as err:
        raise ValueError(f"{activation} is not valid activation.") from err


def round_up(x: int, div: int = 1) -> int:
    return x + (-x % div)


def get_anchors_and_scalers(
    detection_head_output_shape: List[ShapeDtypeStruct], input_size: Tuple[int, int]
) -> Tuple[jax.Array, jax.Array]:
    def get_box_strides(
        detection_head_output_shape: List[ShapeDtypeStruct], input_height: int
    ) -> List[int]:
        strides = []
        for output_shape in detection_head_output_shape:
            strides.append(input_height // output_shape.shape[1])
        return strides

    img_w, img_h = input_size
    strides = get_box_strides(detection_head_output_shape, img_h)
    anchors_list = []
    scalers_list = []
    for stride in strides:
        anchor_num = img_w // stride * img_h // stride
        scalers_list.append(jax.numpy.full((anchor_num,), stride))
        shift = stride // 2
        h = jax.numpy.arange(0, img_h, stride) + shift
        w = jax.numpy.arange(0, img_w, stride) + shift
        anchor_h, anchor_w = jax.numpy.meshgrid(h, w, indexing="ij")
        anchor = jax.numpy.stack(
            [jax.numpy.ravel(anchor_w), jax.numpy.ravel(anchor_h)], axis=1
        )
        anchors_list.append(anchor)
    anchors = jax.lax.concatenate(anchors_list, dimension=0)
    scalers = jax.lax.concatenate(scalers_list, dimension=0)
    return anchors, scalers
