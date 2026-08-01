import subprocess
import sys
from pathlib import Path

def test_supplied_data_preflight():
    software = Path(__file__).resolve().parents[1]
    data = software.parent / "apo_data"
    result = subprocess.run([sys.executable, str(software / "scripts" / "validate_dataset.py"), "--data", str(data)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
