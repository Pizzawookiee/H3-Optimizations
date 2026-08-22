'''Tests for the cross-platform manual Sparse Sage installer entry point.'''

import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch


PACK = Path(__file__).resolve().parents[1]
PRESTARTUP = PACK / 'prestartup_script.py'
SKIP_ENV = 'H3_OPTIMIZATIONS_SKIP_SPARSE_INSTALL'


class PrestartupCliTests(unittest.TestCase):
    def test_direct_cli_reports_unavailable_and_returns_failure(self):
        environment = os.environ.copy()
        environment[SKIP_ENV] = '1'
        result = subprocess.run(
            [sys.executable, str(PRESTARTUP)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            env=environment,
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('automatic Sparse Sage installation is disabled', result.stdout)
        self.assertIn('Sparse Sage is unavailable; see messages above', result.stdout)

    def test_cli_helper_reports_success(self):
        with patch.dict(os.environ, {SKIP_ENV: '1'}, clear=False):
            namespace = runpy.run_path(str(PRESTARTUP))
        run_installer = namespace['run_installer']
        run_installer.__globals__['ensure_sparse_sage'] = lambda: True
        with patch('builtins.print') as printer:
            self.assertTrue(run_installer(cli=True))
        printer.assert_called_once_with('[H3 Optimizations] Sparse Sage is ready')


if __name__ == '__main__':
    unittest.main()
