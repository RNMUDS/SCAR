"""Phase 0 smoke test: verify environment is ready for Phase 1."""
import subprocess
import sys


def test_python_version():
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"


def test_imports():
    import numpy
    import scipy
    import pandas
    import sklearn
    import torch
    import transformers
    import sentence_transformers
    import ollama
    import rank_bm25  # noqa: F401


def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "CUDA must be available on DGX Spark"
    assert "GB10" in torch.cuda.get_device_name(0)


def test_ollama_reachable():
    out = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, timeout=10
    )
    assert out.returncode == 0
    assert "nomic-embed-text" in out.stdout, "nomic-embed-text model required"


def test_disk_space():
    import shutil
    free_gb = shutil.disk_usage("/home").free / 1e9
    assert free_gb > 100, f"Need 100GB+ free, got {free_gb:.1f}GB"


if __name__ == "__main__":
    for fn_name in [
        "test_python_version", "test_imports", "test_cuda_available",
        "test_ollama_reachable", "test_disk_space"
    ]:
        try:
            globals()[fn_name]()
            print(f"PASS  {fn_name}")
        except AssertionError as e:
            print(f"FAIL  {fn_name}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR {fn_name}: {type(e).__name__}: {e}")
            sys.exit(2)
    print("\nAll Phase 0 smoke tests passed.")
