'''CPU contracts for the Stage 0 report reader and its decision tree.'''

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK / 'benchmarks'))

from report_mlp_stage0 import (  # noqa: E402
    load,
    main,
    render,
    verdict,
)


def selection(rows):
    return [
        {
            'selector': name,
            'tile_budget': budget,
            'oracle_mass_captured': captured,
            'oracle_topk_overlap': overlap,
            'samples': 4,
        }
        for name, budget, captured, overlap in rows
    ]


def sharing(rows):
    return [
        {
            'granularity': granularity,
            'groups': groups,
            'delta_mass_captured': captured,
            'fraction_of_token_oracle': ratio,
            'samples': 4,
        }
        for granularity, groups, captured, ratio in rows
    ]


WORKING = {
    'completed': True,
    'error': None,
    'layers': [0, 10],
    'selection_records': 12,
    'cache_records': 8,
    'notes': [],
    'pooled': {
        'selection': selection([
            ('random', 0.5, 0.51, 0.5),
            ('hit_rate', 0.5, 0.73, 0.66),
            ('attention_update_energy', 0.5, 0.84, 0.71),
            ('combined', 0.5, 0.82, 0.70),
            ('oracle', 0.5, 0.96, 1.0),
        ]),
        'group_sharing': sharing([
            ('token', 14, 0.80, 1.0),
            ('tile64', 14, 0.76, 0.95),
            ('tile128', 14, 0.74, 0.92),
            ('video', 14, 0.70, 0.87),
        ]),
        'token_pair_group_overlap': {'group_rank_jaccard_top14': 0.62},
        'delta_decomposition': {
            'modulation_only_fraction': 0.71,
            'state_only_fraction': 0.34,
            'modulation_cosine': 0.83,
            'state_cosine': 0.41,
            'residual_after_modulation_fraction': 0.29,
        },
        'cache': {
            'bfloat16': {
                'output_relative_error': 0.004,
                'activation_delta_relative_error': 0.02,
                'fc2_relative_error': 0.02,
                'group_rank_jaccard_top14': 0.98,
            },
            'float8_e4m3fn': {
                'output_relative_error': 0.026,
                'activation_delta_relative_error': 0.19,
                'fc2_relative_error': 0.18,
                'group_rank_jaccard_top14': 0.81,
            },
        },
    },
}

FLAT = {
    'completed': True,
    'error': None,
    'layers': [0],
    'selection_records': 4,
    'cache_records': 0,
    'notes': ['sparse attention was disabled'],
    'pooled': {
        'selection': selection([
            ('random', 0.5, 0.50, 0.5),
            ('hit_rate', 0.5, 0.52, 0.51),
            ('attention_update_energy', 0.5, 0.51, 0.50),
            ('oracle', 0.5, 0.95, 1.0),
        ]),
        'group_sharing': [],
        'token_pair_group_overlap': {},
        'delta_decomposition': {
            'modulation_only_fraction': None,
            'state_only_fraction': None,
        },
        'cache': {},
    },
}


class RenderTests(unittest.TestCase):
    def test_every_section_is_present(self):
        text = render(WORKING)
        for heading in (
            'A. MLP-update energy captured',
            'B. SwiGLU delta mass captured',
            'C. Where the cross-step SwiGLU delta comes from',
            'D. Relative error of each cache representation',
            'Decision',
        ):
            self.assertIn(heading, text)

    def test_selectors_and_values_are_shown(self):
        text = render(WORKING)
        self.assertIn('hit_rate', text)
        self.assertIn('attention_update_energy', text)
        self.assertIn('73.0%', text)
        self.assertIn('84.0%', text)
        self.assertIn('96.0%', text)

    def test_notes_and_missing_sections_are_reported(self):
        text = render(FLAT)
        self.assertIn('sparse attention was disabled', text)
        self.assertIn('no cache measurements in this run', text)

    def test_render_survives_an_empty_summary(self):
        text = render({'pooled': {}})
        self.assertIn('no sparse route was measured', text)
        self.assertIn('not measured', text)


class VerdictTests(unittest.TestCase):
    def test_a_working_selector_recommends_building_the_arms(self):
        lines = verdict(WORKING['pooled'])
        self.assertIn('build the A arms', lines[1])
        self.assertIn('attention_update_energy', lines[1])

    def test_a_flat_selector_recommends_stopping(self):
        lines = verdict(FLAT['pooled'])
        self.assertIn('do not build the A execution path yet', lines[1])

    def test_shared_group_mask_is_called_out(self):
        lines = verdict(WORKING['pooled'])
        self.assertIn('shared column mask is enough', lines[2])

    def test_per_token_selection_is_called_out_when_sharing_fails(self):
        pooled = dict(WORKING['pooled'])
        pooled['group_sharing'] = sharing([
            ('token', 14, 0.80, 1.0),
            ('video', 14, 0.30, 0.38),
        ])
        lines = verdict(pooled)
        self.assertIn('per-token feature selection is doing real work', lines[2])

    def test_modulation_share_drives_the_c_verdict(self):
        high = verdict(WORKING['pooled'])[3]
        self.assertIn('shared modulation correction', high)
        pooled = dict(WORKING['pooled'])
        pooled['delta_decomposition'] = {'modulation_only_fraction': 0.1}
        self.assertIn('state-driven', verdict(pooled)[3])

    def test_cache_verdict_compares_both_representations(self):
        line = verdict(WORKING['pooled'])[4]
        self.assertIn('0.026', line)
        self.assertIn('0.004', line)

    def test_unmeasured_sections_say_so(self):
        lines = verdict({})
        self.assertEqual(len(lines), 5)
        for line in lines[1:]:
            self.assertIn('not measured', line)


class LoadTests(unittest.TestCase):
    def test_load_and_main_read_a_report_directory(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / 'summary.json').write_text(
                json.dumps(WORKING),
                encoding='utf-8',
            )
            self.assertEqual(load(directory)['layers'], [0, 10])
            self.assertEqual(main([str(directory)]), 0)
            self.assertEqual(main([str(directory), '--json']), 0)


if __name__ == '__main__':
    unittest.main()
