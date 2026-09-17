# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host configuration selection is atomic and never changes process arguments."""

import importlib.util
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest


@pytest.mark.parametrize("field", ["kv_source_layer_ids", "index_source_layer_ids"])
@pytest.mark.parametrize("sources", [(8, 2, 14, 20), (2, 8, 8, 14, 20), (2.0, 8, 14, 20)])
def test_cache_owner_lists_reject_ambiguous_order_and_types(config, field, sources):
    with pytest.raises(ValueError, match="strictly increasing integer"):
        replace(config.FLASH, **{field: sources})


def test_kv_owner_must_publish_its_own_index_cache(config):
    with pytest.raises(ValueError, match="KV sources must also be index sources"):
        replace(config.FLASH, index_source_layer_ids=(2, 14, 20, 24, 28, 32, 36))


def test_flash_layers_resolve_latest_matching_cache_owners(config):
    model = config.FLASH
    for layer in model.backbone_layers():
        for field, actual in (
            ("kv_source_layer_ids", layer.kv_source_layer_id),
            ("index_source_layer_ids", layer.index_source_layer_id),
        ):
            owners = [owner for owner in getattr(model, field)
                      if owner <= layer.layer_id and model.compress_ratios[owner] == layer.compression_ratio]
            assert actual == (max(owners) if owners else None)


@pytest.fixture
def config(monkeypatch):
    # Configuration only needs symbolic dimension declarations, no compiler.
    package = ModuleType("pypto")
    package.__path__ = []
    language = ModuleType("pypto.language")
    language.dynamic = lambda name: name
    package.language = language
    monkeypatch.setitem(sys.modules, "pypto", package)
    monkeypatch.setitem(sys.modules, "pypto.language", language)
    monkeypatch.setattr(sys, "argv", ["host-server", "--tp", "4", "--ep", "8"])
    name = "_v41_configuration_test.config"
    path = Path(__file__).resolve().parents[2] / "models/deepseek_v4_1_flash/config.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_explicit_configuration_updates_all_shards_without_argv_mutation(config):
    before = list(sys.argv)
    config.configure_kernel_parallelism(1, 2)
    assert sys.argv == before
    assert (config.TP_SIZE, config.EP_SIZE, config.DP_SIZE) == (1, 2, 2)
    assert config.LOCAL_H == config.H
    assert config.LOCAL_O_GROUPS == config.O_GROUPS
    assert config.LOCAL_O_WIDTH == config.O_GROUPS * config.O_LORA
    assert config.N_LOCAL_EXPERTS == config.N_EXPERTS // 2
    assert config.PREFILL_RECV_MAX == config.RECV_MAX == 2 * config.PREFILL_MAX_TOKENS
    assert config.DECODE_RECV_MAX == 2 * config.DECODE_MAX_TOKENS


def test_loaded_kernel_rejects_incompatible_configuration_atomically(config, monkeypatch):
    config.configure_kernel_parallelism(1, 2)
    monkeypatch.setitem(sys.modules, "_v41_configuration_test.decode_swa", ModuleType("decode_swa"))
    config.configure_kernel_parallelism(1, 2)
    with pytest.raises(RuntimeError, match="already imported"):
        config.configure_kernel_parallelism(2, 2)
    assert (config.TP_SIZE, config.EP_SIZE, config.LOCAL_H) == (1, 2, config.H)


@pytest.mark.parametrize("tp,ep", [(True, 2), (3, 8), (8, 2), (1, 1)])
def test_invalid_parallelism_leaves_the_original_configuration(config, tp, ep):
    with pytest.raises(ValueError):
        config.configure_kernel_parallelism(tp, ep)
    assert (config.TP_SIZE, config.EP_SIZE) == (4, 8)


def test_host_loader_imports_real_configuration_before_server_argv_validation(monkeypatch):
    """Isolate the import graph as a new host process would, without invoking JIT."""
    prefix = "models.deepseek_v4_1_flash"
    before = {name: value for name, value in sys.modules.items() if name == prefix or name.startswith(prefix + ".")}
    for name in before:
        monkeypatch.delitem(sys.modules, name)
    package = ModuleType(prefix)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "models/deepseek_v4_1_flash")]
    monkeypatch.setitem(sys.modules, prefix, package)
    pypto = ModuleType("pypto")
    language = ModuleType("pypto.language")
    language.dynamic = lambda name: name
    pypto.language = language
    monkeypatch.setitem(sys.modules, "pypto", pypto)
    monkeypatch.setitem(sys.modules, "pypto.language", language)
    monkeypatch.setattr(sys, "argv", ["pypto-serving", "--tp", "1", "--ep", "1"])
    arguments = list(sys.argv)
    try:
        loader = importlib.import_module(prefix + ".local_attention")
        config = loader.load_kernel_configuration(1, 2)
        assert (config.TP_SIZE, config.EP_SIZE, config.LOCAL_H) == (1, 2, config.H)
        assert sys.argv == arguments
        assert loader._INITIAL_PARALLELISM is None
        assert loader.load_kernel_configuration(1, 2) is config
        with pytest.raises(ValueError):
            loader.load_kernel_configuration(1, 1)
        assert loader._INITIAL_PARALLELISM is None
        assert (config.TP_SIZE, config.EP_SIZE) == (1, 2)
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)
        sys.modules.update(before)
