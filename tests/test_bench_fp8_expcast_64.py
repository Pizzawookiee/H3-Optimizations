"""CPU contracts for the 64x64 FP8 experiment benchmark."""

import importlib.util
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[1] / 'benchmarks' / 'bench_fp8_expcast_64.py'
SPEC = importlib.util.spec_from_file_location('bench_fp8_expcast_64', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_benchmark_arms_rotates_launch_order(monkeypatch):
    calls = []
    monkeypatch.setattr(bench.torch.cuda, 'synchronize', lambda: None)

    def event_ms(function):
        value = function()
        return value, float(len(calls))

    monkeypatch.setattr(bench, 'event_ms', event_ms)
    functions = {
        name: lambda name=name: calls.append(name) or name
        for name in ('int8', 'fp8', 'expcast')
    }

    outputs, timings = bench.benchmark_arms(
        functions, warmup=0, iterations=3, retain_outputs=True
    )

    assert calls == [
        'int8', 'fp8', 'expcast',
        'fp8', 'expcast', 'int8',
        'expcast', 'int8', 'fp8',
    ]
    assert outputs == {'int8': 'int8', 'fp8': 'fp8', 'expcast': 'expcast'}
    assert all(len(result['samples_ms']) == 3 for result in timings.values())
