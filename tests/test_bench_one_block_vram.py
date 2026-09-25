"""CPU-only contracts for the API one-block VRAM benchmark."""

from pathlib import Path
import sys
from types import SimpleNamespace


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_one_block_vram import (
    QKV_LINEAR_ONLY_WARNING,
    _stage_ms,
    build_arm_prompt,
    build_free_barrier_prompt,
    is_expected_block_stop,
    render_table,
)


class FakeSchemas:
    async def inputs(self, _node_type, _overrides, links):
        return dict(links)


class PromptSchemas:
    async def inputs(self, _node_type, overrides, links):
        return {**overrides, **links}


def test_expected_stop_accepts_comfy_qualified_exception_name():
    data = {
        "exception_type": (
            r"C:\ComfyUI\custom_nodes\ComfyUI-H3-Extended."
            "h3_block_lab.live.BlockLabCaptureComplete"
        ),
        "exception_message": (
            "H3 VRAM block report saved; diagnostic sampling stopped as requested"
        ),
    }
    assert is_expected_block_stop(data, True)
    assert not is_expected_block_stop(data, False)


def test_expected_stop_rejects_other_failures():
    assert not is_expected_block_stop({
        "exception_type": "BlockLabError",
        "exception_message": "diagnostic sampling stopped as requested",
    }, True)


def test_free_barrier_is_unique_cpu_only_output_graph():
    import asyncio

    first = asyncio.run(build_free_barrier_prompt(FakeSchemas()))
    second = asyncio.run(build_free_barrier_prompt(FakeSchemas()))
    assert set(first) == {"free_barrier"}
    assert first["free_barrier"]["class_type"] == "PreviewAny"
    assert first["free_barrier"]["inputs"]["source"] != second["free_barrier"]["inputs"]["source"]
    assert not is_expected_block_stop({
        "exception_type": "BlockLabCaptureComplete",
        "exception_message": "unrelated failure",
    }, True)


def prompt_args(**overrides):
    values = dict(
        unet="convrot.safetensors",
        clip="clip.safetensors",
        vae="vae.safetensors",
        prompt="test",
        width=1376,
        height=768,
        frames=124,
        run_tag="qkv_timing",
        sampler="res_multistep",
        scheduler="simple",
        schedule_steps=20,
        step=0,
        seed=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qkv_comparison_forces_config0_on_both_arms():
    import asyncio

    args = prompt_args()
    control = asyncio.run(build_arm_prompt(
        PromptSchemas(), "qkv_control_config0", args,
    ))
    optimized = asyncio.run(build_arm_prompt(
        PromptSchemas(), "qkv_optimized", args,
    ))
    control_patches = [
        node for node in control.values()
        if node["_meta"]["title"].startswith("patch")
    ]
    optimized_patches = [
        node for node in optimized.values()
        if node["_meta"]["title"].startswith("patch")
    ]
    assert control_patches[0]["class_type"] == "H3BenchmarkForceQKVConfig0"
    assert control_patches[1]["class_type"] == "H3MemoryOptimization"
    assert control_patches[1]["inputs"]["qkv_streaming_mode"] == "Off"
    assert optimized_patches[0]["class_type"] == "H3BenchmarkForceQKVConfig0"
    assert optimized_patches[1]["class_type"] == "H3MemoryOptimization"
    assert optimized_patches[1]["inputs"]["qkv_streaming_mode"] == "Auto"
    assert _stage_ms({"measurement": {"timing": {"stages": {
        "qkv_linear_only": {"gpu_ms": 12.3456},
    }}}}, "qkv_linear_only") == "12.346"


def test_core_sparse_arm_sets_dynamic_inputs_and_can_skip_memory():
    import asyncio

    args = prompt_args(step=1)
    graph = asyncio.run(build_arm_prompt(
        PromptSchemas(), "core_sol_stock", args,
    ))
    patches = [
        node for node in graph.values()
        if node["_meta"]["title"].startswith("patch")
    ]
    assert [node["class_type"] for node in patches] == [
        "BlockSparseAttention",
        "H3AIMDOResidencyLimiter",
        "MiniMaxH3VRAMBlockStopZi",
    ]
    inputs = patches[0]["inputs"]
    assert inputs["selection"] == "sol-attn"
    assert inputs["selection.tau"] == 1.3
    assert inputs["start_percent"] == 0.0
    assert patches[2]["inputs"]["step"] == 1
    assert graph["split"]["inputs"]["step"] == 2


def test_qkv_linear_output_is_explicitly_non_comparable():
    table = render_table([{
        "label": "diagnostic",
        "peak_mib": 1024,
        "peak_over_baseline_mib": 512,
        "measurement": {"timing": {"stages": {
            "qkv_linear_only": {"gpu_ms": 12.0},
            "block_total": {"gpu_ms": 34.0},
        }}},
        "routes": {},
    }])
    assert "QKV linear-only ms*" in table
    assert QKV_LINEAR_ONLY_WARNING in table
    assert "do not use this column" in table


if __name__ == "__main__":
    test_expected_stop_accepts_comfy_qualified_exception_name()
    test_expected_stop_rejects_other_failures()
    test_free_barrier_is_unique_cpu_only_output_graph()
    test_qkv_comparison_forces_config0_on_both_arms()
    test_core_sparse_arm_sets_dynamic_inputs_and_can_skip_memory()
    test_qkv_linear_output_is_explicitly_non_comparable()
    print("one-block VRAM benchmark: PASS (6 tests)")
