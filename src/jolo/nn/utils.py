from typing import Literal, Tuple, TypeAlias, Union

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
