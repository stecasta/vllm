# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linear layers narrower than mm_mxfp8's minimum must not hard-fail.

MXFP8 kernel selection is global: `init_mxfp8_linear_kernel()` takes no
per-layer shape, so every MXFP8 layer in a model gets the same kernel. A
checkpoint that quantizes a narrow projection to MXFP8 (observed:
Qwen3.8-Flash-Next `linear_attn.in_proj_ba`, out_features=96) therefore lands
on the FlashInfer CUTLASS kernel regardless, and the shape assert in
`apply_weights` fires inside a torch.compile region during `profile_run`,
where Dynamo finds no handler and engine init dies.
"""

import pytest

from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    MXFP8_MIN_DIM,
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)

reason = FlashInferCutlassMxfp8LinearKernel.unsupported_shape_reason


def test_supported_shape_returns_none():
    assert reason(4096, 4096) is None
    assert reason(MXFP8_MIN_DIM, MXFP8_MIN_DIM) is None


def test_narrow_out_features_is_rejected():
    """The observed failure: out_features=96 with a wide K."""
    got = reason(96, 4096)
    assert got is not None
    assert "N=96" in got


def test_narrow_in_features_is_rejected():
    got = reason(4096, 64)
    assert got is not None
    assert "K=64" in got


def test_k_must_be_block_aligned():
    k = MXFP8_MIN_DIM + 1
    assert k % MXFP8_BLOCK_SIZE != 0, "test needs a non-block-aligned K"
    got = reason(4096, k)
    assert got is not None
    assert "block size" in got


@pytest.mark.parametrize("n,k", [(96, 4096), (4096, 64), (96, 96)])
def test_reason_is_reported_not_raised(n, k):
    """The check must return a reason, never raise: callers branch on it."""
    assert isinstance(reason(n, k), str)
