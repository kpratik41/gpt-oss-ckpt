# FlashInfer 0.6.12 wheel parts

These are ordinary Git files, with a maximum part size of 24,000,000 bytes
(24 MB), below 24.5 MB. The smaller wheel uses a single numbered part.
Additional ZIP compression was kept only when it reduced the file size.

## Restore the original wheels

From the repository root, with Python 3 installed:

```sh
python wheel-parts/flashinfer-0.6.12/reassemble.py
```

On systems using the `python3` command, substitute `python3` above.
The two original wheels appear in `wheel-parts/flashinfer-0.6.12/restored`.
The script verifies SHA-256 checksums for every part, the joined payload,
and the original wheel, and unpacks any extra ZIP layer automatically.
No third-party packages are needed for reassembly.

If downloading files individually, keep all numbered parts, `manifest.json`,
and `reassemble.py` together, then run `python reassemble.py` there.
Parts are binary slices and cannot be installed or extracted individually.

## Contents

| Original wheel | Original bytes | Stored bytes | Compression | Parts |
| --- | ---: | ---: | --- | ---: |
| flashinfer_cubin-0.6.12-py3-none-any.whl | 447533460 | 437578659 | zip | 19 |
| flashinfer_python-0.6.12-py3-none-any.whl | 13985243 | 13182892 | zip | 1 |
