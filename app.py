"""Vercel entrypoint: exposes the FastAPI app defined in src/api.py."""
import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# LightGBM needs the OpenMP runtime (libgomp.so.1). Serverless Linux images may not
# include it, so load the bundled copy when the system one is missing.
if sys.platform.startswith("linux"):
    try:
        ctypes.CDLL("libgomp.so.1")
    except OSError:
        ctypes.CDLL(str(ROOT / "vendor" / "libgomp.so.1"), mode=ctypes.RTLD_GLOBAL)

from src.api import app  # noqa: E402,F401
