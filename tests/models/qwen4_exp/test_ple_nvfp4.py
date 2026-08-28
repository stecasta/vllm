# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 PLE n-gram table: gather layout and dequantization.

The checkpoint stores the table as packed uint8 ``[rows, head_dim // 2]`` plus
float8_e4m3 block scales ``[rows, head_dim // 16]``. The offload path
concatenates a gathered row's packed bytes and scale bytes into one uint8
tensor so the cross-process buffer stays a single tensor of a single dtype.
These tests pin that layout and prove the concat/split round-trip does not
perturb the dequantized result.
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ple_layer import (
    NVFP4_PLE_BLOCK_SIZE,
    dequantize_nvfp4_ple_rows,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

HEAD_DIM = 160
ROWS = 64


def _synthetic_table(device: str):
    """Packed weights, block scales and a global scale with checkpoint shapes."""
    generator = torch.Generator(device=device).manual_seed(0)
    packed = torch.randint(
        0, 256, (ROWS, HEAD_DIM // 2), dtype=torch.uint8,
        device=device, generator=generator,
    )
    # Keep scales in a benign range; e4m3 denormals are not what is under test.
    scales = (
        torch.rand(
            ROWS, HEAD_DIM // NVFP4_PLE_BLOCK_SIZE,
            device=device, generator=generator,
        )
        + 0.5
    ).to(torch.float8_e4m3fn)
    global_scale = torch.tensor([0.37], dtype=torch.float32, device=device)
    return packed, scales, global_scale


def _gather(packed, scales, ids):
    """Mirror Qwen4ExpPLENvfp4EmbeddingMethod.embedding."""
    return torch.cat(
        (
            torch.nn.functional.embedding(ids, packed),
            torch.nn.functional.embedding(ids, scales.view(torch.uint8)),
        ),
        dim=-1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_concat_split_roundtrip_matches_direct_dequant():
    """Splitting the concatenated row must equal dequantizing the parts."""
    device = "cuda"
    packed, scales, global_scale = _synthetic_table(device)
    ids = torch.arange(ROWS, device=device)

    rows = _gather(packed, scales, ids)
    assert rows.dtype is torch.uint8
    assert rows.shape[-1] == HEAD_DIM // 2 + HEAD_DIM // NVFP4_PLE_BLOCK_SIZE

    got = dequantize_nvfp4_ple_rows(rows, HEAD_DIM, global_scale, torch.bfloat16)
    want = dequantize_to_dtype(
        packed, scales, global_scale, torch.bfloat16,
        block_size=NVFP4_PLE_BLOCK_SIZE, swizzle=False,
    )
    assert got.shape == want.shape == (ROWS, HEAD_DIM)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_gather_permutation_is_row_exact():
    """A permuted gather must dequantize to the permuted reference rows."""
    device = "cuda"
    packed, scales, global_scale = _synthetic_table(device)
    ids = torch.randperm(ROWS, device=device)

    got = dequantize_nvfp4_ple_rows(
        _gather(packed, scales, ids), HEAD_DIM, global_scale, torch.bfloat16
    )
    full = dequantize_to_dtype(
        packed, scales, global_scale, torch.bfloat16,
        block_size=NVFP4_PLE_BLOCK_SIZE, swizzle=False,
    )
    torch.testing.assert_close(got, full[ids], rtol=0, atol=0)


def test_row_width_is_validated():
    """A row narrower than packed+scale bytes is a hard error, not a silent slice."""
    too_narrow = torch.zeros(4, HEAD_DIM // 2, dtype=torch.uint8)
    with pytest.raises(ValueError, match="smaller than"):
        dequantize_nvfp4_ple_rows(
            too_narrow, HEAD_DIM, torch.tensor([1.0]), torch.bfloat16
        )


def test_head_dim_must_be_block_aligned():
    """create_weights rejects an embedding dim that is not block aligned."""
    from vllm.models.qwen4_exp.nvidia.ple_layer import (
        Qwen4ExpPLENvfp4EmbeddingMethod,
    )

    with pytest.raises(ValueError, match="multiple of"):
        Qwen4ExpPLENvfp4EmbeddingMethod().create_weights(
            torch.nn.Module(), HEAD_DIM + 1, [8], 0, 0, torch.bfloat16
        )
