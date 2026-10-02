# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen4Exp model components."""

from .hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

# Transformers >= 5.18 renamed "qwen_sparse_attention" to "indexed_attention".
QWEN4_EXP_ATTENTION_LAYER_TYPES = frozenset(
    {"qwen_sparse_attention", "indexed_attention"}
)

__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "QWEN4_EXP_ATTENTION_LAYER_TYPES",
]
