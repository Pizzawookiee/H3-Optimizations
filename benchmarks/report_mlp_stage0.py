'''Turn one Stage 0 report directory into the four decision tables.

    python benchmarks/report_mlp_stage0.py output/h3_mlp_stage0/<run>

Reads only the JSON the run already wrote, so it needs no GPU and no ComfyUI.
'''

import argparse
import json
from pathlib import Path
import sys

# Below this the attention selectors are not buying anything over the control.
RANDOM_MARGIN = 0.05
# Above this a shared group mask is close enough to per-token selection.
SHARED_MASK_MARGIN = 0.9


def _percent(value):
    if value is None or value != value:
        return '   n/a'
    return '%5.1f%%' % (100.0 * value)


def _ratio(value):
    if value is None or value != value:
        return '  n/a'
    return '%5.3f' % value


def load(directory):
    directory = Path(directory)
    summary = json.loads(
        (directory / 'summary.json').read_text(encoding='utf-8')
    )
    return summary


def selection_table(pooled):
    rows = pooled.get('selection') or []
    if not rows:
        return ['A. attention selection: no sparse route was measured']
    budgets = sorted({row['tile_budget'] for row in rows})
    by_key = {(row['selector'], row['tile_budget']): row for row in rows}
    selectors = [
        name for name in (
            'random',
            'hit_rate',
            'attention_update_energy',
            'combined',
            'oracle',
        )
        if any(key[0] == name for key in by_key)
    ]
    lines = ['A. MLP-update energy captured by each selector top-k']
    for budget in budgets:
        lines.append('')
        lines.append(
            '   %s of target-video KV tiles retained' % _percent(budget)
        )
        for name in selectors:
            row = by_key.get((name, budget))
            if row is None:
                continue
            lines.append(
                '     %-24s captures %s   oracle overlap %s'
                % (
                    name,
                    _percent(row['oracle_mass_captured']),
                    _ratio(row['oracle_topk_overlap']),
                )
            )
    return lines


def group_table(pooled):
    rows = pooled.get('group_sharing') or []
    if not rows:
        return ['B. delta structure: no cache measurements in this run']
    budgets = sorted({row['groups'] for row in rows})
    by_key = {(row['granularity'], row['groups']): row for row in rows}
    lines = ['B. SwiGLU delta mass captured by a top-k ConvRot group mask']
    for count in budgets:
        lines.append('')
        lines.append('   %d of 56 groups' % count)
        for granularity in ('token', 'tile64', 'tile128', 'video'):
            row = by_key.get((granularity, count))
            if row is None:
                continue
            lines.append(
                '     %-10s captures %s   of per-token oracle %s'
                % (
                    granularity,
                    _percent(row['delta_mass_captured']),
                    _ratio(row['fraction_of_token_oracle']),
                )
            )
    overlap = pooled.get('token_pair_group_overlap') or {}
    if overlap:
        lines.append('')
        lines.append('   mean top-k group overlap between token pairs')
        for field, value in sorted(overlap.items()):
            lines.append('     %-28s %s' % (field, _ratio(value)))
    return lines


def modulation_table(pooled):
    decomposition = pooled.get('delta_decomposition') or {}
    if not any(value is not None for value in decomposition.values()):
        return ['C. AdaLN share: no cache measurements in this run']
    lines = ['C. Where the cross-step SwiGLU delta comes from']
    for field in (
        'modulation_only_fraction',
        'state_only_fraction',
        'modulation_cosine',
        'state_cosine',
        'residual_after_modulation_fraction',
    ):
        lines.append('   %-38s %s' % (field, _ratio(decomposition.get(field))))
    return lines


def cache_table(pooled):
    cache = pooled.get('cache') or {}
    measured = {
        dtype: entry for dtype, entry in cache.items()
        if entry.get('output_relative_error') is not None
    }
    if not measured:
        return ['D. cache representation: no cache measurements in this run']
    lines = ['D. Relative error of each cache representation']
    fields = [
        'output_relative_error',
        'activation_delta_relative_error',
        'fc2_relative_error',
    ]
    for dtype, entry in sorted(measured.items()):
        lines.append('   %s' % dtype)
        for field in fields:
            lines.append('     %-34s %s' % (field, _ratio(entry.get(field))))
        for field in sorted(entry):
            if field.startswith('group_rank_jaccard'):
                lines.append('     %-34s %s' % (field, _ratio(entry[field])))
    return lines


def verdict(pooled):
    '''The short decision tree, stated against what the run measured.'''
    lines = ['Decision']
    rows = pooled.get('selection') or []
    by_key = {(row['selector'], row['tile_budget']): row for row in rows}
    budgets = sorted({row['tile_budget'] for row in rows})
    gains = []
    for budget in budgets:
        control = by_key.get(('random', budget))
        if control is None:
            continue
        for name in ('hit_rate', 'attention_update_energy', 'combined'):
            row = by_key.get((name, budget))
            if row is None:
                continue
            gains.append((
                row['oracle_mass_captured'] - control['oracle_mass_captured'],
                name,
                budget,
            ))
    if not gains:
        lines.append('   A: not measured')
    else:
        best, name, budget = max(gains)
        if best < RANDOM_MARGIN:
            lines.append(
                '   A: attention selectors beat random by at most %s; do not '
                'build the A execution path yet' % _percent(best)
            )
        else:
            lines.append(
                '   A: %s beats random by %s at a %s budget; build the A arms'
                % (name, _percent(best), _percent(budget))
            )

    shared = [
        row for row in (pooled.get('group_sharing') or [])
        if row['granularity'] != 'token'
        and row['fraction_of_token_oracle'] is not None
    ]
    if not shared:
        lines.append('   B: not measured')
    else:
        best = max(shared, key=lambda row: row['fraction_of_token_oracle'])
        if best['fraction_of_token_oracle'] >= SHARED_MASK_MARGIN:
            lines.append(
                '   B: a %s-shared %d-group mask keeps %s of the per-token '
                'oracle; a shared column mask is enough'
                % (
                    best['granularity'],
                    best['groups'],
                    _ratio(best['fraction_of_token_oracle']),
                )
            )
        else:
            lines.append(
                '   B: the best shared mask keeps only %s of the per-token '
                'oracle; per-token feature selection is doing real work'
                % _ratio(best['fraction_of_token_oracle'])
            )

    decomposition = pooled.get('delta_decomposition') or {}
    modulation = decomposition.get('modulation_only_fraction')
    if modulation is None:
        lines.append('   C: not measured')
    elif modulation >= 0.5:
        lines.append(
            '   C: AdaLN modulation accounts for %s of the delta; a shared '
            'modulation correction should come before per-token work'
            % _percent(modulation)
        )
    else:
        lines.append(
            '   C: AdaLN modulation accounts for only %s of the delta; the '
            'delta is state-driven' % _percent(modulation)
        )

    cache = pooled.get('cache') or {}
    fp8 = cache.get('float8_e4m3fn') or {}
    if fp8.get('output_relative_error') is None:
        lines.append('   D: not measured')
    else:
        lines.append(
            '   D: FP8 costs %s on the cached output and %s on the delta; '
            'BF16 costs %s and %s'
            % (
                _ratio(fp8.get('output_relative_error')),
                _ratio(fp8.get('activation_delta_relative_error')),
                _ratio(
                    (cache.get('bfloat16') or {}).get('output_relative_error')
                ),
                _ratio(
                    (cache.get('bfloat16') or {}).get(
                        'activation_delta_relative_error'
                    )
                ),
            )
        )
    return lines


def render(summary):
    pooled = summary.get('pooled') or {}
    lines = [
        'H3 MLP Stage 0 report',
        '  completed        %s' % summary.get('completed'),
        '  layers           %s' % ','.join(
            str(layer) for layer in summary.get('layers', ())
        ),
        '  selection records %d' % summary.get('selection_records', 0),
        '  cache records     %d' % summary.get('cache_records', 0),
    ]
    if summary.get('error'):
        lines.append('  error            %s' % summary['error'])
    for note in summary.get('notes', ()):
        lines.append('  note             %s' % note)
    for section in (
        selection_table(pooled),
        group_table(pooled),
        modulation_table(pooled),
        cache_table(pooled),
        verdict(pooled),
    ):
        lines.append('')
        lines.extend(section)
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument(
        '--json',
        action='store_true',
        help='print the pooled summary as JSON instead of tables',
    )
    args = parser.parse_args(argv)
    summary = load(args.directory)
    if args.json:
        print(json.dumps(summary.get('pooled', {}), indent=2, sort_keys=True))
    else:
        print(render(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
