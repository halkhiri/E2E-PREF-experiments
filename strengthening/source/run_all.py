"""Compatibility entry point for the final resumable experimental protocol."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('run_parallel.py')),run_name='__main__')
