from __future__ import annotations

import unittest

from tools.extract_hidden_sglang import _supervise_processes


class _Process:
    def __init__(self, exitcodes):
        self._exitcodes = iter(exitcodes)
        self._last = None
        self.terminated = False
        self.joined = False

    @property
    def exitcode(self):
        try:
            self._last = next(self._exitcodes)
        except StopIteration:
            pass
        return self._last

    def is_alive(self):
        return self.exitcode is None and not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        del timeout
        self.joined = True


class StageBSupervisionTest(unittest.TestCase):
    def test_first_failed_rank_terminates_live_siblings(self):
        failed = _Process([1])
        sibling = _Process([None, None, None])
        with self.assertRaisesRegex(RuntimeError, "rank workers failed"):
            _supervise_processes([failed, sibling], poll_seconds=0)
        self.assertTrue(sibling.terminated)
        self.assertTrue(failed.joined)
        self.assertTrue(sibling.joined)

    def test_all_clean_ranks_return(self):
        first = _Process([None, 0])
        second = _Process([0])
        _supervise_processes([first, second], poll_seconds=0)
        self.assertTrue(first.joined)
        self.assertTrue(second.joined)


if __name__ == "__main__":
    unittest.main()
