# causal-conv1d 1.7.0: CUDA 13.0 / PyTorch 2.11

Astral-built wheel for CPython 3.12 and Linux x86_64:

`causal_conv1d-1.7.0+cu.13.0.torch.2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`

Source: [Astral release v1.7.0](https://github.com/astral-sh-build/build-causal-conv1d/releases/tag/v1.7.0).
The original download matches the SHA-256 digest published for that release.

The 206,786,335-byte wheel is split without extra compression into 9 binary
parts. The first 8 are 24,000,000 bytes each; the last is 14,786,335 bytes.
Every part is strictly below 24.5 MB and is stored as an ordinary Git file.

## Restore the wheel

From the repository root:

```sh
python wheel-parts/causal-conv1d-cu13-torch2.11/reassemble.py
```

Use `python3` if needed. The original wheel is written to
`wheel-parts/causal-conv1d-cu13-torch2.11/restored/`. The script checks the
size and SHA-256 digest of every part and of the reconstructed wheel.
It needs no third-party Python packages.

For individual downloads, keep all 9 parts, `manifest.json`, and
`reassemble.py` together, then run `python reassemble.py` there. Individual
parts cannot be installed with pip. Install only the reconstructed wheel
in an environment matching its Python, CUDA, PyTorch, and platform tags.
