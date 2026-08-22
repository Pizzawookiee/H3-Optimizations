'''Create merge/error curves and layer-step heatmaps from a Stage 1 report.'''

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_ERROR_BUDGETS = (0.001, 0.003, 0.01, 0.03, 0.1)


def _load_rows(report_directory):
    path = Path(report_directory) / 'layer_step_metrics.jsonl'
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError('the report contains no layer-step metrics')
    return rows


def _slug(value):
    return ('%.6g' % float(value)).replace('.', 'p').replace('-', 'm')


def _write_curve_csv(rows, output):
    fields = (
        'selector',
        'reconstruction',
        'target_merge_fraction',
        'merge_fraction_median',
        'residual_median',
        'residual_p95',
        'residual_p99',
        'block_median',
        'block_p95',
        'block_p99',
    )
    grouped = {}
    for row in rows:
        key = (
            row['selector'],
            row['reconstruction'],
            float(row['target_merge_fraction']),
        )
        grouped.setdefault(key, []).append(row)
    curve_rows = []
    for key, samples in sorted(grouped.items()):
        curve_rows.append({
            'selector': key[0],
            'reconstruction': key[1],
            'target_merge_fraction': key[2],
            'merge_fraction_median': float(np.median([
                sample['merge_fraction'] for sample in samples
            ])),
            'residual_median': float(np.median([
                sample['residual_error']['median'] for sample in samples
            ])),
            'residual_p95': float(np.median([
                sample['residual_error']['p95'] for sample in samples
            ])),
            'residual_p99': float(np.median([
                sample['residual_error']['p99'] for sample in samples
            ])),
            'block_median': float(np.median([
                sample['block_error']['median'] for sample in samples
            ])),
            'block_p95': float(np.median([
                sample['block_error']['p95'] for sample in samples
            ])),
            'block_p99': float(np.median([
                sample['block_error']['p99'] for sample in samples
            ])),
        })
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curve_rows)
    return curve_rows


def _plot_curves(curves, output):
    import matplotlib.pyplot as plt

    selectors = sorted({row['selector'] for row in curves})
    reconstructions = sorted({row['reconstruction'] for row in curves})
    figure, axes = plt.subplots(
        len(reconstructions),
        len(selectors),
        figsize=(4.2 * len(selectors), 3.8 * len(reconstructions)),
        squeeze=False,
        sharex=True,
    )
    for row_index, reconstruction in enumerate(reconstructions):
        for column_index, selector in enumerate(selectors):
            axis = axes[row_index, column_index]
            selected = sorted(
                (
                    row for row in curves
                    if row['selector'] == selector
                    and row['reconstruction'] == reconstruction
                ),
                key=lambda row: row['merge_fraction_median'],
            )
            x = [row['merge_fraction_median'] * 100.0 for row in selected]
            axis.plot(x, [row['residual_median'] for row in selected], label='median')
            axis.plot(x, [row['residual_p95'] for row in selected], label='p95')
            axis.plot(x, [row['residual_p99'] for row in selected], label='p99')
            axis.set_yscale('log')
            axis.grid(True, alpha=0.25)
            axis.set_title('%s\n%s' % (selector, reconstruction))
            axis.set_xlabel('MLP row reduction (%)')
            if column_index == 0:
                axis.set_ylabel('Gated residual relative error')
            if row_index == 0 and column_index == 0:
                axis.legend()
    figure.suptitle('H3 target-video 1T x 2Y x 2X MLP sharing oracle')
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _heatmap(rows, selector, reconstruction, budget):
    selected = [
        row for row in rows
        if row['selector'] == selector
        and row['reconstruction'] == reconstruction
    ]
    layers = sorted({int(row['layer']) for row in selected})
    steps = sorted({int(row['step_index']) for row in selected})
    layer_index = {value: index for index, value in enumerate(layers)}
    step_index = {value: index for index, value in enumerate(steps)}
    matrix = np.zeros((len(layers), len(steps)), dtype=np.float64)
    for row in selected:
        p95 = row['residual_error']['p95']
        if p95 is not None and float(p95) <= float(budget):
            y = layer_index[int(row['layer'])]
            x = step_index[int(row['step_index'])]
            matrix[y, x] = max(matrix[y, x], float(row['merge_fraction']))
    return layers, steps, matrix


def _write_heatmap_csv(layers, steps, matrix, output):
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['layer'] + steps)
        for layer, values in zip(layers, matrix):
            writer.writerow([layer] + [float(value) for value in values])


def _plot_heatmap(layers, steps, matrix, title, output):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(max(8, len(steps) * 0.3), 5.5))
    image = axis.imshow(matrix * 100.0, aspect='auto', vmin=0.0, vmax=50.0)
    axis.set_yticks(range(len(layers)), labels=layers)
    if len(steps) <= 30:
        axis.set_xticks(range(len(steps)), labels=steps, rotation=90)
    else:
        stride = max(1, len(steps) // 20)
        ticks = list(range(0, len(steps), stride))
        axis.set_xticks(ticks, labels=[steps[index] for index in ticks])
    axis.set_xlabel('Diffusion step')
    axis.set_ylabel('H3 layer')
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label('Mergeable target-video tokens (%)')
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def analyze(report_directory, error_budgets=DEFAULT_ERROR_BUDGETS):
    report_directory = Path(report_directory)
    rows = _load_rows(report_directory)
    curves = _write_curve_csv(
        rows,
        report_directory / 'merge_fraction_vs_error.csv',
    )
    _plot_curves(curves, report_directory / 'merge_fraction_vs_error.png')

    selectors = sorted({row['selector'] for row in rows})
    reconstructions = sorted({row['reconstruction'] for row in rows})
    outputs = []
    for selector in selectors:
        for reconstruction in reconstructions:
            for budget in error_budgets:
                layers, steps, matrix = _heatmap(
                    rows,
                    selector,
                    reconstruction,
                    budget,
                )
                name = 'heatmap_%s_%s_p95_%s' % (
                    selector,
                    reconstruction,
                    _slug(budget),
                )
                csv_path = report_directory / (name + '.csv')
                png_path = report_directory / (name + '.png')
                _write_heatmap_csv(layers, steps, matrix, csv_path)
                _plot_heatmap(
                    layers,
                    steps,
                    matrix,
                    '%s / %s: p95 gated error <= %.4g'
                    % (selector, reconstruction, budget),
                    png_path,
                )
                outputs.extend((csv_path, png_path))
    return [
        report_directory / 'merge_fraction_vs_error.csv',
        report_directory / 'merge_fraction_vs_error.png',
        *outputs,
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('report_directory')
    parser.add_argument(
        '--error-budgets',
        default=','.join(str(value) for value in DEFAULT_ERROR_BUDGETS),
    )
    args = parser.parse_args(argv)
    budgets = tuple(
        float(value.strip())
        for value in args.error_budgets.split(',')
        if value.strip()
    )
    outputs = analyze(args.report_directory, budgets)
    print(json.dumps({'outputs': [str(path) for path in outputs]}, indent=2))


if __name__ == '__main__':
    main()
