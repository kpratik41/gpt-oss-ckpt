# cuda-tile and tileiras wheels

- `cuda_tile-1.8.0-cp312-cp312-manylinux2014_x86_64.whl` is uploaded unchanged (534,490 bytes).
- `nvidia_cuda_tileiras-13.5.37-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` is split into two binary parts (24,000,000 and 18,173,556 bytes), with no extra compression.

Both source wheels were verified against the SHA-256 values published by PyPI.
Every uploaded part is strictly below 24.5 MB.

## Restore tileiras

From the repository root:

```sh
python wheel-parts/cuda-tile/reassemble.py
```

Use `python3` if needed. The script requires only the Python standard library,
checks every part and the final wheel using SHA-256, and writes the original
tileiras wheel to `wheel-parts/cuda-tile/restored/`.

If downloading individually, keep both numbered parts, `manifest.json`, and
`reassemble.py` together, then run `python reassemble.py` in that directory.
The cuda-tile wheel needs no reassembly. Do not pass `.part001` or `.part002`
directly to pip; they are slices of a wheel, not standalone wheel files.

These files target Linux x86_64; cuda-tile uses CPython 3.12. The two wheels
are not a complete offline dependency bundle for `cuda-tile[tileiras]`.
