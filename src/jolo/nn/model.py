from typing import Sequence, Tuple

import jax
from flax import nnx
from flax import typing as flax_typing

from jolo.nn import blocks


class Yolo(nnx.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        head_bias_init: Sequence[float],
        dtype: flax_typing.Dtype,
        rngs: nnx.Rngs,
    ):
        self.b1_conv = blocks.ConvBlock(
            in_features=3,
            out_features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=dtype,
            rngs=rngs,
        )
        self.b2_conv = blocks.ConvBlock(
            in_features=self.b1_conv.out_features,
            out_features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=dtype,
            rngs=rngs,
        )
        self.b2_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.b2_conv.out_features,
            out_features=256,
            part_features=128,
            dtype=dtype,
            rngs=rngs,
        )

        self.b3_adown = blocks.ADown(
            in_features=self.b2_rep_ncspelan.out_features,
            out_features=256,
            dtype=dtype,
            rngs=rngs,
        )
        self.b3_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.b3_adown.out_features,
            out_features=512,
            part_features=256,
            dtype=dtype,
            rngs=rngs,
        )

        self.b4_adown = blocks.ADown(
            in_features=self.b3_rep_ncspelan.out_features,
            out_features=512,
            dtype=dtype,
            rngs=rngs,
        )
        self.b4_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.b4_adown.out_features,
            out_features=512,
            part_features=512,
            dtype=dtype,
            rngs=rngs,
        )

        self.b5_adown = blocks.ADown(
            in_features=self.b4_rep_ncspelan.out_features,
            out_features=512,
            dtype=dtype,
            rngs=rngs,
        )
        self.b5_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.b5_adown.out_features,
            out_features=512,
            part_features=512,
            dtype=dtype,
            rngs=rngs,
        )

        self.n3_sspelan = blocks.SPPELAN(
            in_features=self.b5_rep_ncspelan.out_features,
            out_features=512,
            dtype=dtype,
            rngs=rngs,
        )

        self.n4_upsample = blocks.UpSample(
            in_features=self.n3_sspelan.out_features,
            scale_factor=(2, 2),
            method="nearest",
        )
        self.n4_concat = blocks.Concatenate(
            [self.n4_upsample.out_features, self.b4_rep_ncspelan.out_features]
        )
        self.n4_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.n4_concat.out_features,
            out_features=512,
            part_features=512,
            dtype=dtype,
            rngs=rngs,
        )

        self.p3_upsample = blocks.UpSample(
            in_features=self.n4_rep_ncspelan.out_features,
            scale_factor=(2, 2),
            method="nearest",
        )
        self.p3_concat = blocks.Concatenate(
            in_features_list=[
                self.p3_upsample.out_features,
                self.b3_rep_ncspelan.out_features,
            ]
        )

        self.p3_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.p3_concat.out_features,
            out_features=256,
            part_features=256,
            dtype=dtype,
            rngs=rngs,
        )

        self.p4_adown = blocks.ADown(
            in_features=self.p3_rep_ncspelan.out_features,
            out_features=256,
            dtype=dtype,
            rngs=rngs,
        )
        self.p4_concat = blocks.Concatenate(
            [self.p4_adown.out_features, self.n4_rep_ncspelan.out_features]
        )
        self.p4_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.p4_concat.out_features,
            out_features=512,
            part_features=512,
            dtype=dtype,
            rngs=rngs,
        )

        self.p5_adown = blocks.ADown(
            in_features=self.p4_rep_ncspelan.out_features,
            out_features=512,
            dtype=dtype,
            rngs=rngs,
        )
        self.p5_concat = blocks.Concatenate(
            in_features_list=[
                self.p5_adown.out_features,
                self.n3_sspelan.out_features,
            ]
        )
        self.p5_rep_ncspelan = blocks.RepNCSPELAN(
            in_features=self.p5_concat.out_features,
            out_features=512,
            part_features=512,
            dtype=dtype,
            rngs=rngs,
        )

        self.mulithead_detection = blocks.MultiHeadDetection(
            in_features_list=[
                self.p3_rep_ncspelan.out_features,
                self.p4_rep_ncspelan.out_features,
                self.p5_rep_ncspelan.out_features,
            ],
            num_classes=num_classes,
            bias_init=head_bias_init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array
    ) -> Tuple[Sequence[jax.Array], Sequence[jax.Array], Sequence[jax.Array]]:
        x = self.b1_conv(x)

        x = self.b2_conv(x)
        x = self.b2_rep_ncspelan(x)

        x = self.b3_adown(x)
        b3_rep_ncspelan = x = self.b3_rep_ncspelan(x)

        x = self.b4_adown(x)
        b4_rep_ncspelan = x = self.b4_rep_ncspelan(x)

        x = self.b5_adown(x)
        x = self.b5_rep_ncspelan(x)

        n3_sppelan = x = self.n3_sspelan(x)

        x = self.n4_upsample(x)
        x = self.n4_concat([x, b4_rep_ncspelan])
        n4_rep_ncspelan = x = self.n4_rep_ncspelan(x)

        x = self.p3_upsample(x)
        x = self.p3_concat([x, b3_rep_ncspelan])
        p3_rep_ncspelan = x = self.p3_rep_ncspelan(x)

        x = self.p4_adown(x)
        x = self.p4_concat([x, n4_rep_ncspelan])
        p4_rep_ncspelan = x = self.p4_rep_ncspelan(x)

        x = self.p5_adown(x)
        x = self.p5_concat([x, n3_sppelan])
        p5_rep_ncspelan = x = self.p5_rep_ncspelan(x)

        return self.mulithead_detection(
            [p3_rep_ncspelan, p4_rep_ncspelan, p5_rep_ncspelan]
        )
