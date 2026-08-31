import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loc_count


class LocCountTests(unittest.TestCase):
    def test_skips_cmake_build_trees(self) -> None:
        self.assertIn("build", loc_count.SKIP_DIRS)
        self.assertIn("CMakeFiles", loc_count.SKIP_DIRS)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "ok.rs").write_text("fn x() {}\n", encoding="utf-8")
            generated = root / "oxidize-cpp" / "build" / "CMakeFiles"
            generated.mkdir(parents=True)
            (generated / "junk.cpp").write_text("// generated\n" * 50, encoding="utf-8")
            data = loc_count.count(root)
            self.assertEqual(data["total"], 1)
            self.assertEqual(data["by_lang"]["rust"], 1)
