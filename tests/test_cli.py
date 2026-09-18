"""Unit tests for the governed_agent CLI dispatcher."""
import os
import unittest
from governed_agent.cli import main


class TestCLI(unittest.TestCase):

    def test_cli_help(self):
        # Should return 0 when run with help
        self.assertEqual(main(["--help"]), 0)

    def test_cli_triage(self):
        ret = main(["triage", "--reward", "0.02", "--loss", "0.10", "--prior", "0.50"])
        self.assertEqual(ret, 0)

    def test_cli_audit(self):
        # Audit this test file itself
        this_file = os.path.abspath(__file__)
        ret = main(["audit", this_file, "--max-mutants", "3"])
    def test_cli_info(self):
        ret = main(["info"])
        self.assertEqual(ret, 0)

    def test_cli_gate_help(self):
        ret = main(["gate", "--help"])
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    unittest.main()
