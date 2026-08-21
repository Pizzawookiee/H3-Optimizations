'''Static ownership checks for the standalone repository.'''

from pathlib import Path
import re
import unittest

PACK = Path(__file__).resolve().parents[1]
SOURCE = PACK / 'h3_optimizations'


class SourceIsolationTests(unittest.TestCase):
    def test_python_source_has_no_experimental_pack_dependencies(self):
        banned = (
            'ComfyUI-H3-Extended',
            'h3_attention',
            'h3_activation_memory',
            'h3_runtime',
            'h3_probe',
            'epilogue',
            'minimax_h3::',
            'torch.no_grad',
            'torch.inference_mode',
            'torch.cuda.empty_cache',
            'torch.cuda.synchronize',
        )
        for path in SOURCE.rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            for fragment in banned:
                self.assertNotIn(fragment, text, '%s contains %s' % (path, fragment))

    def test_every_custom_op_uses_the_repo_namespace(self):
        declarations = []
        for path in SOURCE.rglob('*.py'):
            text = path.read_text(encoding='utf-8')
            for match in re.finditer(r'custom_op\(', text):
                declarations.append(text[match.start():match.start() + 200])
        self.assertTrue(declarations)
        for declaration in declarations:
            self.assertIn('h3_optimizations::', declaration)

    def test_package_metadata_points_to_the_canonical_repo(self):
        metadata = (PACK / 'pyproject.toml').read_text(encoding='utf-8')
        self.assertIn(
            'https://github.com/Zironic/H3-Optimizations',
            metadata,
        )

    def test_dense_kitchen_integration_uses_only_public_carrier_apis(self):
        source = (SOURCE / 'kitchen_qkv.py').read_text(encoding='utf-8')
        self.assertNotIn('._C', source)
        self.assertNotIn('PrequantizedInt8Attention(', source)

    def test_legacy_dense_stack_is_not_exported_to_production(self):
        production_boundaries = (
            SOURCE / 'attention' / '__init__.py',
            SOURCE / 'qkv' / '__init__.py',
            SOURCE / 'qkv' / 'projectors.py',
            SOURCE / 'apply.py',
        )
        banned = (
            'SM89SageMemoryEfficientBackend',
            'DenseFusedQKVProjector',
            'sage_mem_eff',
            'dense_fused_qkv',
        )
        for path in production_boundaries:
            text = path.read_text(encoding='utf-8')
            for fragment in banned:
                self.assertNotIn(fragment, text, '%s exports %s' % (path, fragment))


if __name__ == '__main__':
    unittest.main()
