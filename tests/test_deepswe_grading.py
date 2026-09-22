"""Offline checks for the standalone DeepSWE grading adapter."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from si2ca.runtime import deepswe_grading

NUM_GPUS = 0


class DeepSWEGrading(unittest.IsolatedAsyncioTestCase):
    async def test_assets_are_copied_and_reward_details_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / 'grader.py'
            asset.write_bytes(b'grader fixture')
            reward = {'reward': 1, 'f2p_total': 2, 'f2p_passed': 2, 'p2p_total': 3, 'p2p_passed': 3}
            sandbox = SimpleNamespace(
                exec=AsyncMock(side_effect=[(0, '', '')] * 3 + [(0, json.dumps(reward), '')]),
                write_file=AsyncMock(),
            )
            metadata = {
                'workdir': '/workspace',
                'eval_files': {'/tests/grader.py': str(asset)},
                'eval_cmd': 'python /tests/grader.py > /logs/verifier/test_stdout.log',
                'deepswe': {'verifier_timeout_sec': 90},
            }
            result = await deepswe_grading.grade(sandbox, metadata)
        self.assertEqual(result, (1.0, 'OK', {k: v for k, v in reward.items() if k != 'reward'}))
        sandbox.write_file.assert_awaited_once_with('/tests/grader.py', b'grader fixture', user='root')
        sandbox.exec.assert_any_await(
            'cd /workspace && python /tests/grader.py > /logs/verifier/test-stdout.txt',
            user='root', check=False, timeout=210,
        )

    async def test_zero_reward_is_a_completed_grade(self):
        sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[
            (0, '', ''), (1, '', 'test failed'), (0, '{"reward": 0}', ''),
        ]))
        self.assertEqual(await deepswe_grading.grade(sandbox, {'eval_cmd': 'verify'}), (0.0, 'OK', {}))

    async def test_missing_or_invalid_reward_is_an_infrastructure_failure(self):
        for reward, status in [('', 'GRADEFAIL:NO_REWARD_JSON:'), ('not json', 'GRADEFAIL:BAD_JSON:')]:
            with self.subTest(reward=reward):
                sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[
                    (0, '', ''), (1, '', 'verifier failed'), (0, reward, ''),
                ]))
                score, label, detail = await deepswe_grading.grade(sandbox, {'eval_cmd': 'verify'})
                self.assertEqual((score, detail), (0.0, {}))
                self.assertTrue(label.startswith(status), label)

    async def test_verifier_timeout_does_not_read_a_stale_reward(self):
        sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[(0, '', ''), (1, '', 'timeout')]))
        clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 1801]))
        with patch.object(deepswe_grading, 'time', clock):
            result = await deepswe_grading.grade(sandbox, {'eval_cmd': 'verify'})
        self.assertEqual(result, (0.0, 'TIMEOUT:1801s', {}))
        self.assertEqual(sandbox.exec.await_count, 2)

    async def test_missing_or_uncopyable_assets_do_not_run_the_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / 'grader.py'
            for exists, status in [(False, 'GRADEFAIL:ASSET_MISSING:'), (True, 'GRADEFAIL:COPY:')]:
                with self.subTest(exists=exists):
                    if exists:
                        asset.write_bytes(b'grader fixture')
                    sandbox = SimpleNamespace(
                        exec=AsyncMock(return_value=(0, '', '')),
                        write_file=AsyncMock(side_effect=OSError('copy failed')),
                    )
                    score, label, detail = await deepswe_grading.grade(sandbox, {
                        'eval_cmd': 'verify', 'eval_files': {'/tests/grader.py': str(asset)},
                    })
                    self.assertEqual((score, detail), (0.0, {}))
                    self.assertTrue(label.startswith(status), label)
                    self.assertFalse(any('verify' == call.args[0] or '&& verify' in call.args[0]
                                         for call in sandbox.exec.await_args_list))


if __name__ == '__main__':
    unittest.main()
