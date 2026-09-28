"""CPU contracts for the all-layer one-step profile driver."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import aiohttp

import bench_one_block_vram
from bench_one_block_vram import BenchError, run_prompt
from bench_one_step_profile import ARMS, build_arm_prompt


class PromptSchemas:
    async def inputs(self, _node_type, overrides, links):
        return {**overrides, **links}


def args(**overrides):
    values = dict(
        unet="convrot.safetensors",
        conditioning_tokens=1450,
        conditioning_cache_entry="conditioning.safetensors",
        width=1376,
        height=768,
        frames=124,
        run_tag="one_step",
        sampler="res_multistep",
        scheduler="simple",
        schedule_steps=20,
        allocator_snapshot=False,
        seed=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def patch_nodes(graph):
    return [
        node for node in graph.values()
        if node["_meta"]["title"].startswith("patch")
    ]


def test_default_matrix_separates_memory_and_attention_costs():
    assert tuple(ARMS) == (
        "kitchen_baseline",
        "kitchen_memory",
        "native_kv100",
        "native_kv30",
        "native_kv30_lower_vram",
    )


def test_native_arm_is_fixed_kv100_and_profiles_all_layers():
    graph = asyncio.run(build_arm_prompt(
        PromptSchemas(), "native_kv100", args(allocator_snapshot=True),
    ))
    sparse = next(
        node for node in patch_nodes(graph)
        if node["class_type"] == "H3SparseAttentionAdvanced"
    )
    assert sparse["inputs"]["video_budget"] == 1.0
    assert sparse["inputs"]["early_kv"] == 1.0
    assert sparse["inputs"]["late_kv"] == 1.0
    profile = graph["profile"]
    assert profile["class_type"] == "MiniMaxH3OneStepProfileZi"
    assert profile["inputs"]["allocator_snapshot"] is True
    assert graph["split"]["inputs"]["step"] == 1


def test_dense_memory_arm_only_adds_memory_patch():
    baseline = asyncio.run(build_arm_prompt(
        PromptSchemas(), "kitchen_baseline", args(),
    ))
    memory = asyncio.run(build_arm_prompt(
        PromptSchemas(), "kitchen_memory", args(),
    ))
    assert [node["class_type"] for node in patch_nodes(baseline)] == [
        "H3BenchmarkForceQKVConfig0",
        "ModelAttentionBackend",
    ]
    assert [node["class_type"] for node in patch_nodes(memory)] == [
        "H3BenchmarkForceQKVConfig0",
        "ModelAttentionBackend",
        "H3MemoryOptimization",
    ]


def test_lower_vram_arm_only_changes_v_staging():
    standard = asyncio.run(build_arm_prompt(
        PromptSchemas(), "native_kv30", args(),
    ))
    lower = asyncio.run(build_arm_prompt(
        PromptSchemas(), "native_kv30_lower_vram", args(),
    ))
    standard_patches = patch_nodes(standard)
    lower_patches = patch_nodes(lower)
    assert [node["class_type"] for node in standard_patches] == [
        node["class_type"] for node in lower_patches
    ]
    standard_memory = next(
        node for node in standard_patches
        if node["class_type"] == "H3MemoryOptimization"
    )
    lower_memory = next(
        node for node in lower_patches
        if node["class_type"] == "H3MemoryOptimization"
    )
    assert "kitchen_v_memory_mode" not in standard_memory["inputs"]
    assert lower_memory["inputs"]["kitchen_v_memory_mode"] == "Lower VRAM (slower)"
    for node_type in ("H3SparseAttentionAdvanced",):
        standard_sparse = next(
            node for node in standard_patches if node["class_type"] == node_type
        )
        lower_sparse = next(
            node for node in lower_patches if node["class_type"] == node_type
        )
        assert standard_sparse["inputs"] == lower_sparse["inputs"]
        assert standard_sparse["inputs"]["video_budget"] == 0.3
        assert standard_sparse["inputs"]["early_kv"] == 0.3
        assert standard_sparse["inputs"]["late_kv"] == 0.3


def test_clean_interrupt_is_an_opt_in_expected_stop():
    class FakeWebSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def receive(self):
            return SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps({
                    "type": "execution_interrupted",
                    "data": {
                        "prompt_id": "profile-prompt",
                        "node_id": "sample",
                        "node_type": "SamplerCustomAdvanced",
                    },
                }),
            )

    class FakeSession:
        def ws_connect(self, _url):
            return FakeWebSocket()

    async def fake_queue_prompt(*_args):
        return "profile-prompt"

    original_queue_prompt = bench_one_block_vram.queue_prompt
    bench_one_block_vram.queue_prompt = fake_queue_prompt
    try:
        result = asyncio.run(run_prompt(
            FakeSession(), "http://127.0.0.1:8188", "client", {}, 1.0, True,
            expect_interrupt=True,
        ))
        assert result["expected_stop"] is True
        assert result["exception"]["type"] == "InterruptProcessingException"

        try:
            asyncio.run(run_prompt(
                FakeSession(), "http://127.0.0.1:8188", "client", {}, 1.0, True,
            ))
        except BenchError as exc:
            assert str(exc) == "execution interrupted"
        else:
            raise AssertionError("ordinary interruptions must remain failures")
    finally:
        bench_one_block_vram.queue_prompt = original_queue_prompt


if __name__ == "__main__":
    test_default_matrix_separates_memory_and_attention_costs()
    test_native_arm_is_fixed_kv100_and_profiles_all_layers()
    test_dense_memory_arm_only_adds_memory_patch()
    test_lower_vram_arm_only_changes_v_staging()
    test_clean_interrupt_is_an_opt_in_expected_stop()
    print("one-step profile benchmark: PASS (5 tests)")
