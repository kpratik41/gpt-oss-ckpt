# Mamba 2.3.2.post1: CUDA 13.0 / PyTorch 2.11

Astral-built wheel for CPython 3.12 and Linux x86_64:

`mamba_ssm-2.3.2.post1+cu.13.0.torch.2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`

Source: [Astral release v2.3.2.post1-r2](https://github.com/astral-sh-build/build-mamba/releases/tag/v2.3.2.post1-r2).
The downloaded original matches the SHA-256 digest published for that release.

The 349,371,542-byte wheel is split without extra compression into 15 binary
parts. The first 14 are 24,000,000 bytes each; the last is 13,371,542 bytes.
Every part is strictly below 24.5 MB and is stored as an ordinary Git file.

## Restore the wheel

From the repository root:

```sh
python wheel-parts/mamba-cu13-torch2.11/reassemble.py
```

Use `python3` if needed. The restored wheel is written to
`wheel-parts/mamba-cu13-torch2.11/restored/`. The script checks the size and
SHA-256 digest of every part and of the reconstructed wheel. It needs no
third-party Python packages.

For individual downloads, keep all 15 numbered parts, `manifest.json`, and
`reassemble.py` together, then run `python reassemble.py` there. Individual
parts cannot be installed with pip. Install only the reconstructed `.whl`
in an environment matching its Python, CUDA, PyTorch, and platform tags.
