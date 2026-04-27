from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_success.data.loaders import validate_raw_layout


class LoaderTests(unittest.TestCase):
    def test_validate_raw_layout_reports_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            Path(raw_dir, "student-por.csv").write_text("G3\n10\n", encoding="utf-8")
            status = validate_raw_layout(raw_dir)

        self.assertTrue(status["student-por.csv"])
        self.assertFalse(status["student-mat.csv"])


if __name__ == "__main__":
    unittest.main()
