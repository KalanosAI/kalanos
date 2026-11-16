"""Offline notebook checks; these are not a live grader or Colab test."""

import ast
import json
import tempfile
import types
import unittest
from pathlib import Path


NB = json.loads(
    Path(__file__).with_name("kalanos_synthetic_quickstart.ipynb").read_text()
)
CODES = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]


class NotebookTests(unittest.TestCase):
    def test_all_cells_compile(self):
        for cell in CODES:
            compile(cell, "<notebook>", "exec")

    def test_download_script_compiles(self):
        tree = ast.parse(CODES[1])
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "download_script" for t in n.targets
            )
        )
        compile(ast.literal_eval(node.value), "<download>", "exec")

    def check_grade(self, exit_code, write_report):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run(command, **kwargs):
                self.assertNotIn("--publish", command)
                if write_report:
                    Path(command[command.index("--report") + 1]).write_text("{}")
                return types.SimpleNamespace(
                    returncode=exit_code, stdout="stub only", stderr="stub"
                )

            ns = dict(
                OUTPUT=root,
                KALANOS=Path("/fake/kalanos"),
                DATASET=Path("/fake/data"),
                subprocess=types.SimpleNamespace(run=run),
                grader_env={},
            )
            fn = next(
                n for n in ast.parse(CODES[2]).body if isinstance(n, ast.FunctionDef)
            )
            exec(
                compile(ast.Module(body=[fn], type_ignores=[]), "<function>", "exec"),
                ns,
            )
            return ns["grade_report"]("json")

    def test_gate_pass_retains_report(self):
        self.check_grade(0, True)

    def test_gate_failure_retains_report(self):
        self.check_grade(1, True)

    def test_operational_failure_raises(self):
        with self.assertRaises(RuntimeError):
            self.check_grade(2, True)

    def test_missing_report_raises(self):
        with self.assertRaises(RuntimeError):
            self.check_grade(0, False)


if __name__ == "__main__":
    unittest.main()
