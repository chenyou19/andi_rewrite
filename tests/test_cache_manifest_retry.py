import json
import os
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from andi_rewrite.engine.evaluation_cache import DiskEvaluationCache


class ManifestRetryTest(unittest.TestCase):
    def test_transient_and_persistent_permission_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DiskEvaluationCache(directory, raw_fingerprint='r', score_fingerprint='s')
            replace = os.replace
            attempts = []

            def transient(source, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    raise PermissionError('busy')
                replace(source, destination)

            cache.manifest['test'] = 1
            with patch('andi_rewrite.engine.evaluation_cache.os.replace', side_effect=transient), patch('andi_rewrite.engine.evaluation_cache.time.sleep'):
                cache._write_manifest()
            self.assertEqual(len(attempts), 3)
            cache.manifest['test'] = 2
            with patch('andi_rewrite.engine.evaluation_cache.os.replace', side_effect=PermissionError('busy')) as mock, patch('andi_rewrite.engine.evaluation_cache.time.sleep'):
                with self.assertRaises(PermissionError):
                    cache._write_manifest()
                self.assertEqual(mock.call_count, 8)
            self.assertEqual(json.loads(cache.manifest_path.read_text())['test'], 1)
