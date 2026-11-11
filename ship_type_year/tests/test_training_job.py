import unittest
import tempfile
from pathlib import Path
from training_job import start_regression,verify_worker

class WorkerTests(unittest.TestCase):
    def test_spawned_worker_completes_real_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'verified.txt';verify_worker(p)
            self.assertTrue(p.exists())
    def test_worker_reports_invalid_input(self):
        process,updates=start_regression('/missing/naval-input.xlsx',10)
        try:
            msg=updates.get(timeout=30)
            self.assertEqual(msg[0],'error');self.assertTrue(msg[1])
        finally:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join(timeout=5)
            updates.close()
