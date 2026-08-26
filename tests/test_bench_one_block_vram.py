"""CPU-only contracts for the API one-block VRAM benchmark."""

from pathlib import Path
import sys


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_one_block_vram import build_free_barrier_prompt, is_expected_block_stop


class FakeSchemas:
    async def inputs(self, _node_type, _overrides, links):
        return dict(links)


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


if __name__ == "__main__":
    test_expected_stop_accepts_comfy_qualified_exception_name()
    test_expected_stop_rejects_other_failures()
    test_free_barrier_is_unique_cpu_only_output_graph()
    print("one-block VRAM benchmark: PASS (3 tests)")
