'''Regression coverage for ComfyUI custom-node package loading.'''

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


PACK = Path(__file__).resolve().parents[1]


class PackageImportIdentityTests(unittest.TestCase):
    def test_comfy_fallback_keeps_one_plan_class_identity(self):
        script = textwrap.dedent(
            r'''
            import importlib.util
            from pathlib import Path
            import sys
            import types

            pack = Path(sys.argv[1]).resolve()
            sys.path = [
                entry
                for entry in sys.path
                if Path(entry or '.').resolve() != pack
            ]

            test_args = sys.argv[2:]
            sys.argv = [sys.argv[0], '--cpu']
            import comfy.options
            comfy.options.enable_args_parsing()

            outer_name = '_synthetic_comfy_custom_nodes'
            outer = types.ModuleType(outer_name)
            outer.__path__ = [str(pack.parent)]
            sys.modules[outer_name] = outer

            node_name = outer_name + '.H3_Optimizations'
            spec = importlib.util.spec_from_file_location(
                node_name,
                pack / '__init__.py',
                submodule_search_locations=[str(pack)],
            )
            node_pack = importlib.util.module_from_spec(spec)
            sys.modules[node_name] = node_pack
            spec.loader.exec_module(node_pack)

            import h3_optimizations.apply as apply
            import h3_optimizations.apply_policy as apply_policy
            import h3_optimizations.memory_migration_node as memory_node
            import h3_optimizations.plan as plan

            assert apply.H3OptimizationPlan is plan.H3OptimizationPlan
            assert (
                memory_node.read_plan.__globals__['H3OptimizationPlan']
                is plan.H3OptimizationPlan
            )
            assert apply_policy._base is apply

            duplicate_names = [
                name
                for name, module in sys.modules.items()
                if name.endswith('.h3_optimizations.plan')
                and module is not plan
            ]
            assert not duplicate_names, duplicate_names

            sys.argv = [sys.argv[0], *test_args]
            '''
        )
        result = subprocess.run(
            [sys.executable, '-c', script, str(PACK), *sys.argv[1:]],
            cwd=PACK.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg='stdout:\n%s\nstderr:\n%s' % (result.stdout, result.stderr),
        )


if __name__ == '__main__':
    unittest.main()
