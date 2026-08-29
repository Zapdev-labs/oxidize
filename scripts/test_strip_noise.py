import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "strip_noise_comments", Path(__file__).with_name("strip_noise_comments.py")
)
_STRIP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STRIP)


class HangEnd(unittest.TestCase):
    def test_hanging_clause_starters_are_truncated(self):
        for fragment in (
            "Resolve a repo_id + filename to a HuggingFace download URL. If",
            "Format an SSE event into `buf` (cap bytes) per the wire format. Handles",
            "Format the registry as a JSON array into `buf` (cap bytes). Returns",
        ):
            self.assertTrue(
                _STRIP.is_truncated_comment(fragment),
                msg=fragment,
            )

    def test_complete_contract_is_kept(self):
        text = (
            "Copy up to `max` shard assignments for `tensor_name` into `out_array`. "
            "Pass NULL for `out_array` to count only."
        )
        self.assertFalse(_STRIP.is_truncated_comment(text))


if __name__ == "__main__":
    unittest.main()
