from typing import Dict, List, Optional, Sequence, Tuple

import jax
from flax import nnx
from flax import typing as flax_typing

from jolo.nn import utils


class ConvBlock(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        kernel_size: utils.DoubleIntTuple,
        strides: utils.DoubleIntTuple = (1, 1),
        feature_group_count: int = 1,
        use_bias: bool = False,
        pad: bool = True,
        activation: Optional[utils.Activation] = "silu",
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        padding: utils.PaddingLike = "valid"
        if pad:
            padding = utils.auto_pad(kernel_size, 1)

        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
            feature_group_count=feature_group_count,
            dtype=dtype,
            rngs=rngs,
        )
        self.batchnorm = nnx.BatchNorm(
            num_features=out_features, momentum=0.9, dtype=dtype, rngs=rngs
        )
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.pad = pad
        self.activation = activation.lower() if activation else activation

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation:
            x = utils.get_activation(self.activation)(x)
        return x


class RepConv(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        kernel_size: utils.DoubleIntTuple = (3, 3),
        activation: Optional[utils.Activation] = "silu",
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        self.conv_block = ConvBlock(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            dtype=dtype,
            activation=None,
            rngs=rngs,
        )

        self.pw_conv_block = ConvBlock(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1, 1),
            dtype=dtype,
            activation=None,
            rngs=rngs,
        )

        self.activation = activation.lower() if activation else activation
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.activation = activation
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        x1 = self.conv_block(x)
        x2 = self.pw_conv_block(x)
        x = x1 + x2
        if self.activation:
            x = utils.get_activation(self.activation)(x)
        return x


class BottleNeck(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        kernel_size: tuple[utils.DoubleIntTuple, utils.DoubleIntTuple] = (
            (3, 3),
            (3, 3),
        ),
        residual: bool = True,
        expand: float = 1.0,
        activation: Optional[utils.Activation] = "silu",
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        neck_features = int(out_features * expand)
        self.rep_conv = RepConv(
            in_features=in_features,
            out_features=neck_features,
            kernel_size=kernel_size[0],
            dtype=dtype,
            rngs=rngs,
        )

        self.conv_block = ConvBlock(
            in_features=self.rep_conv.out_features,
            out_features=out_features,
            kernel_size=kernel_size[1],
            dtype=dtype,
            rngs=rngs,
        )

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.residual = residual
        self.expand = expand
        self.activation = activation
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        input_x = x
        x = self.rep_conv(x)
        x = self.conv_block(x)

        if self.residual:
            if self.in_features != self.out_features:
                raise ValueError(
                    f"Residual connection can't be used: in_features ({self.in_features}) !=",
                    f"out_features ({self.out_features})",
                )
            x = input_x + x

        return x


class RepNCSP(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        kernel_size: utils.DoubleIntTuple = (1, 1),
        csp_expand: float = 0.5,
        repeat: int = 1,
        neck_args: Optional[Dict] = None,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        neck_features = int(out_features * csp_expand)
        self.conv_block_1 = ConvBlock(
            in_features=in_features,
            out_features=neck_features,
            kernel_size=kernel_size,
            dtype=dtype,
            rngs=rngs,
        )
        self.bottleneck_layers = [
            BottleNeck(
                in_features=neck_features,
                out_features=neck_features,
                dtype=dtype,
                rngs=rngs,
                **(neck_args if neck_args else {}),
            )
            for _ in range(repeat)
        ]

        self.conv_block_2 = ConvBlock(
            in_features=in_features,
            out_features=neck_features,
            kernel_size=kernel_size,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv_block_3 = ConvBlock(
            in_features=neck_features + neck_features,
            out_features=out_features,
            kernel_size=kernel_size,
            dtype=dtype,
            rngs=rngs,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.csp_expand = csp_expand
        self.repeat = repeat
        self.neck_args = neck_args
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        x1 = self.conv_block_1(x)

        for bottleneck in self.bottleneck_layers:
            x1 = bottleneck(x1)

        x2 = self.conv_block_2(x)

        x = jax.lax.concatenate([x1, x2], dimension=3)
        x = self.conv_block_3(x)

        return x


class RepNCSPELAN(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        part_features: int,
        process_features: Optional[int] = None,
        csp_args: Optional[Dict] = None,
        csp_neck_args: Optional[Dict] = None,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        if not process_features:
            process_features = part_features // 2

        self.conv_block_1 = ConvBlock(
            in_features=in_features,
            out_features=part_features,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs,
        )
        self.rep_ncsp_1 = RepNCSP(
            in_features=part_features // 2,
            out_features=process_features,
            neck_args=csp_neck_args,
            dtype=dtype,
            rngs=rngs,
            **(csp_args if csp_args else {}),
        )
        self.conv_block_2 = ConvBlock(
            in_features=process_features,
            out_features=process_features,
            kernel_size=(3, 3),
            dtype=dtype,
            rngs=rngs,
        )
        self.rep_ncsp_2 = RepNCSP(
            in_features=process_features,
            out_features=process_features,
            neck_args=csp_neck_args,
            dtype=dtype,
            rngs=rngs,
            **(csp_args if csp_args else {}),
        )
        self.conv_block_3 = ConvBlock(
            in_features=process_features,
            out_features=process_features,
            kernel_size=(3, 3),
            dtype=dtype,
            rngs=rngs,
        )

        self.conv_block_4 = ConvBlock(
            in_features=(part_features + (process_features * 2)),
            out_features=out_features,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.part_features = part_features
        self.process_features = process_features
        self.csp_args = csp_args
        self.csp_neck_args = csp_neck_args
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_block_1(x)

        x1, x2 = jax.numpy.split(x, 2, axis=-1)

        x3 = self.rep_ncsp_1(x2)
        x3 = self.conv_block_2(x3)

        x4 = self.rep_ncsp_2(x3)
        x4 = self.conv_block_3(x3)

        x = jax.lax.concatenate([x1, x2, x3, x4], dimension=3)
        x = self.conv_block_4(x)
        return x


class ELAN(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        part_features: int,
        process_features: Optional[int] = None,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        if not process_features:
            process_features = part_features // 2

        self.conv_block_1 = ConvBlock(
            in_features=in_features,
            out_features=part_features,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs,
        )
        self.conv_block_2 = ConvBlock(
            in_features=part_features // 2,
            out_features=process_features,
            kernel_size=(3, 3),
            dtype=dtype,
            rngs=rngs,
        )
        self.conv_block_3 = ConvBlock(
            in_features=process_features,
            out_features=process_features,
            kernel_size=(3, 3),
            dtype=dtype,
            rngs=rngs,
        )

        self.conv_block_4 = ConvBlock(
            in_features=(part_features + (process_features * 2)),
            out_features=out_features,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.part_features = part_features
        self.process_features = process_features
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_block_1(x)

        x1, x2 = jax.numpy.split(x, 2, axis=-1)

        x3 = self.conv_block_2(x2)

        x4 = self.conv_block_3(x3)

        x = jax.lax.concatenate([x1, x2, x3, x4], dimension=3)
        x = self.conv_block_4(x)
        return x


class PoolBlock(nnx.Module):
    def __init__(
        self,
        max_pool: bool = True,
        window_shape: utils.DoubleIntTuple = (2, 2),
        strides: Optional[utils.DoubleIntTuple] = None,
        pad: bool = True,
    ):
        padding: str | utils.DoubleIntTuple = "VALID"
        if pad:
            padding = utils.auto_pad(window_shape)
        self.max_pool = max_pool
        self.window_shape = window_shape
        self.strides = strides
        self.pad = pad
        self.padding = padding
        self.pool_fn = nnx.max_pool if self.max_pool else nnx.avg_pool

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pool_fn(
            x, self.window_shape, self.strides, [self.padding, self.padding]
        )  # type: ignore
        return x


class SPPELAN(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        neck_features: Optional[int] = None,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        if not neck_features:
            neck_features = out_features // 2

        self.conv_block_1 = ConvBlock(
            in_features=in_features,
            out_features=neck_features,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs,
        )
        self.poolers = [
            PoolBlock(window_shape=(5, 5), strides=(1, 1)) for i in range(3)
        ]
        self.conv_block_2 = ConvBlock(
            in_features=neck_features * 4,
            out_features=out_features,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs,
        )

        self.in_features = in_features
        self.out_features = out_features
        self.neck_features = neck_features
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        features = [self.conv_block_1(x)]

        for pooler in self.poolers:
            features.append(pooler(features[-1]))

        x = jax.lax.concatenate(features, dimension=3)
        x = self.conv_block_2(x)
        return x


class ADown(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        self.pool_block_1 = PoolBlock(
            max_pool=False, window_shape=(2, 2), strides=(1, 1)
        )
        self.conv_block_1 = ConvBlock(
            in_features=in_features // 2,
            out_features=out_features // 2,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=dtype,
            rngs=rngs,
        )
        self.pool_block_2 = PoolBlock(window_shape=(3, 3), strides=(2, 2))
        self.conv_block_2 = ConvBlock(
            in_features=in_features // 2,
            out_features=out_features // 2,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs,
        )

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pool_block_1(x)
        x1, x2 = jax.numpy.split(x, 2, axis=-1)
        x1 = self.conv_block_1(x1)
        x2 = self.pool_block_2(x2)
        x2 = self.conv_block_2(x2)
        return jax.lax.concatenate([x1, x2], dimension=3)


class UpSample(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        scale_factor: utils.DoubleIntTuple,
        method: str = "nearest",
    ):
        self.scale_factor = scale_factor
        self.method = method
        self.in_features = in_features
        self.out_features = in_features

    def __call__(self, x: jax.Array) -> jax.Array:
        b, h, w, c = x.shape
        x = jax.image.resize(
            x,
            (b, h * self.scale_factor[0], w * self.scale_factor[1], c),
            method=self.method,
        )
        return x


# TODO: Accept in_features_list and set out_features
class Concatenate(nnx.Module):
    def __init__(self, in_features_list: Sequence[int], axis=3):
        self.axis = axis
        self.in_features_list = in_features_list
        self.out_features = sum(in_features_list)

    def __call__(self, x: Sequence[jax.Array]) -> jax.Array:
        return jax.lax.concatenate(x, dimension=self.axis)


class ConvSequence(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        inter_features: int,
        out_features: int,
        bias_init: float,
        feature_group_count: int = 1,
        activation: Optional[utils.Activation] = None,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        self.conv_block_1 = ConvBlock(
            in_features=in_features,
            out_features=inter_features,
            kernel_size=(3, 3),
            dtype=dtype,
            rngs=rngs,
        )
        self.conv_block_2 = ConvBlock(
            in_features=inter_features,
            out_features=inter_features,
            kernel_size=(3, 3),
            feature_group_count=feature_group_count,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv = nnx.Conv(
            in_features=inter_features,
            out_features=out_features,
            kernel_size=(1, 1),
            bias_init=nnx.initializers.constant(bias_init),
            feature_group_count=feature_group_count,
            dtype=dtype,
            rngs=rngs,
        )

        self.in_features = in_features
        self.inter_features = inter_features
        self.out_features = out_features
        self.bias_init = bias_init
        self.feature_group_count = feature_group_count
        self.activation = activation.lower() if activation else activation
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv(x)
        if self.activation:
            x = utils.get_activation(self.activation)(x)
        return x


class Detection(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        anchor_inter_features: int,
        class_inter_features: int,
        bias_init: float,
        reg_max: int = 16,
        anchor_feature_group_count: int = 1,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        anchor_features = 4 * reg_max

        self.anchor_conv = ConvSequence(
            in_features=in_features,
            inter_features=anchor_inter_features,
            out_features=anchor_features,
            feature_group_count=anchor_feature_group_count,
            bias_init=1.0,
            dtype=dtype,
            rngs=rngs,
        )
        self.class_conv = ConvSequence(
            in_features=in_features,
            inter_features=class_inter_features,
            out_features=num_classes,
            bias_init=bias_init,
            dtype=dtype,
            rngs=rngs,
        )
        self.in_features = in_features
        self.num_classes = num_classes
        self.anchor_inter_features = anchor_inter_features
        self.class_inter_features = class_inter_features
        self.bias_init = bias_init
        self.reg_max = reg_max
        self.anchor_feature_group_count = anchor_feature_group_count
        self.dtype = dtype

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        anchor_x = self.anchor_conv(x)
        class_x = self.class_conv(x)
        anchor_x, vector_x = self.anc2vec(anchor_x)
        return class_x, anchor_x, vector_x

    def anc2vec(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        b, h, w, c = x.shape
        x_reshaped = jax.lax.reshape(x, (b * h * w, 4, c // 4))
        vector_x = nnx.softmax(x_reshaped, -1) @ jax.numpy.arange(
            0, self.reg_max, dtype=self.dtype
        )
        # vector_x = jax.numpy.sum(vector_x, axis=-1)
        vector_x = jax.lax.reshape(vector_x, (b, h, w, 4))
        return x, vector_x


class MultiHeadDetection(nnx.Module):
    def __init__(
        self,
        *,
        in_features_list: Sequence[int],
        num_classes: int,
        bias_init: int | Sequence[int] = -10,
        reg_max: int | Sequence[int] = 16,
        use_group: bool = True,
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        if not isinstance(reg_max, Sequence):
            reg_max = [reg_max] * len(in_features_list)

        if not isinstance(bias_init, Sequence):
            bias_init = [bias_init] * len(in_features_list)

        self.reg_max = reg_max
        self.bias_init = bias_init

        assert len(self.reg_max) == len(in_features_list)
        assert len(self.bias_init) == len(in_features_list)

        min_in_feature = min(in_features_list)
        anchor_feature_group_count = 4 if use_group else 1

        self.detection_heads: Sequence[Detection] = []
        for i, (r, b) in enumerate(zip(self.reg_max, self.bias_init)):
            anchor_inter_features = max(
                utils.round_up(min_in_feature // 4, anchor_feature_group_count),
                4 * r,
                16,
            )
            class_inter_features = max(min_in_feature, min(num_classes * 2, 128))
            self.detection_heads.append(
                Detection(
                    in_features=in_features_list[i],
                    num_classes=num_classes,
                    anchor_inter_features=anchor_inter_features,
                    class_inter_features=class_inter_features,
                    bias_init=b,
                    reg_max=r,
                    anchor_feature_group_count=anchor_feature_group_count,
                    dtype=dtype,
                    rngs=rngs,
                )
            )

    def __call__(
        self, x: Sequence[jax.Array]
    ) -> Tuple[List[jax.Array], List[jax.Array], List[jax.Array]]:
        classes_list: List[jax.Array] = []
        anchors_list: List[jax.Array] = []
        vectors_list: List[jax.Array] = []
        for i, detection_head in enumerate(self.detection_heads):
            class_x, anchor_x, vector_x = detection_head(x[i])
            classes_list.append(class_x)
            anchors_list.append(anchor_x)
            vectors_list.append(vector_x)

        return classes_list, anchors_list, vectors_list
