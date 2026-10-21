from __future__ import annotations

import ast
import functools
import importlib.util
import sys
import types

import torch
from torch import nn
from torch.nn import functional as functional

from .common import ROOT, file_hash


UPSTREAM = ROOT / "third_party/comparison_upstream"


def load_definitions(path, module_name, *, stop_before=None, exclude_imports=(), include_names=None, rewrites=None):
    if module_name in sys.modules:
        return sys.modules[module_name]
    parsed = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = []
    for node in parsed.body:
        if stop_before is not None and getattr(node, "name", None) == stop_before:
            break
        if include_names is not None:
            if getattr(node, "name", None) in include_names:
                selected.append(node)
            continue
        if isinstance(node, ast.ImportFrom):
            if any((node.module or "").startswith(prefix) for prefix in exclude_imports):
                continue
            if rewrites and node.module in rewrites:
                node.module = rewrites[node.module]
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("__name__"):
            continue
        selected.append(node)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update({"torch": torch, "nn": nn})
    sys.modules[module_name] = module
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@functools.lru_cache(maxsize=1)
def time_weaver_core():
    return load_definitions(UPSTREAM / "contsg/contsg/models/timeweaver.py", "_faultcrafter_timeweaver",
                            stop_before="TimeWeaverModule", exclude_imports=("contsg", "linear_attention_transformer")).TimeWeaverCore


@functools.lru_cache(maxsize=1)
def flow_transformer():
    folder = UPSTREAM / "flowts/FMTS/Models/interpretable_diffusion"
    utilities = load_definitions(folder / "model_utils.py", "_faultcrafter_flow_utilities")
    module = load_definitions(folder / "transformer.py", "_faultcrafter_flow_transformer",
                              rewrites={"Models.interpretable_diffusion.model_utils": utilities.__name__})

    def attention_forward(layer, inputs, encoder_output=None, mask=None):
        context = inputs if encoder_output is None else encoder_output
        batch, length, width = inputs.shape
        keys = layer.key(context)
        queries = layer.query(inputs)
        keys = layer.k_norm(keys) + 0.1 * keys
        queries = layer.q_norm(queries) + 0.1 * queries
        keys = keys.reshape(batch, context.shape[1], layer.n_head, width // layer.n_head).transpose(1, 2)
        queries = queries.reshape(batch, length, layer.n_head, width // layer.n_head).transpose(1, 2)
        values = layer.value(context).reshape(batch, context.shape[1], layer.n_head, width // layer.n_head).transpose(1, 2)
        if encoder_output is not None:
            frequencies = layer.freqs_cis.to(inputs.device)[:length]
            queries, keys = module.apply_rotary_emb(queries.transpose(1, 2), keys.transpose(1, 2), freqs_cis=frequencies)
            queries, keys = queries.transpose(1, 2), keys.transpose(1, 2)
        outputs = functional.scaled_dot_product_attention(queries, keys, values,
                    dropout_p=layer.attn_drop.p if layer.training else 0.0)
        outputs = outputs.transpose(1, 2).reshape(batch, length, width)
        return layer.resid_drop(layer.proj(outputs)), None

    def self_attention(layer, inputs, mask=None):
        return attention_forward(layer, inputs, mask=mask)

    module.FullAttention.forward = self_attention
    module.CrossAttention.forward = attention_forward
    return module.Transformer


@functools.lru_cache(maxsize=1)
def matrix_edm():
    folder = UPSTREAM / "iccdm/iCCDM"
    utilities = load_definitions(folder / "utils.py", "_faultcrafter_iccdm_utilities",
                  include_names={"default", "exists", "identity", "unnormalize_to_zero_to_one",
                                 "normalize_to_neg_one_to_one", "prob_mask_like", "my_log"})
    module = load_definitions(folder / "diffusion.py", "_faultcrafter_iccdm_diffusion", rewrites={"utils": utilities.__name__})
    module.tqdm = lambda iterable, *arguments, **keywords: iterable
    return module.ElucidatedDiffusion


def provenance(method):
    entries = {
        "diffucd": [ROOT / "research/benchmarks/method_sources.json"],
        "ccgan": [UPSTREAM / "ccgan/RC-49/RC-49_64x64/CcGAN-improved/models/cont_cond_cnn_generator_discriminator.py",
                  ROOT / "third_party/comparison_sources/ccgan/RC-49/RC-49_64x64/CcGAN-improved/Train_CcGAN.py"],
        "iccdm": [UPSTREAM / "iccdm/iCCDM/diffusion.py", UPSTREAM / "iccdm/iCCDM/label_embedding.py"],
        "time_weaver": [UPSTREAM / "contsg/contsg/models/timeweaver.py"],
        "flowts": [UPSTREAM / "flowts/FMTS/Models/interpretable_diffusion/transformer.py",
                   UPSTREAM / "flowts/FMTS/Models/interpretable_diffusion/model_utils.py",
                   UPSTREAM / "flowts/FMTS/Models/interpretable_diffusion/FMTS.py"],
    }
    return {str(path.relative_to(ROOT)): file_hash(path) for path in entries[method]}
