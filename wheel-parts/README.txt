Split wheels for kpratik41/gpt-oss-ckpt

Each .partNNN file is at most 24,000,000 bytes (24.0 decimal MB),
strictly below 24.5 MB and GitHub's 25 MiB browser upload limit.

causal_conv1d: 9 parts. ZIP compression reduced 196,558,813 bytes to
196,441,030 bytes (117,783 bytes saved, approximately 0.06%).
mamba_ssm: 14 parts. No additional compression; the tested ZIP was
322,355,097 bytes, larger than the original 322,288,410 bytes.

RESTORE
Keep all 23 parts, manifest.json, and reassemble.py in the same directory.
Run with Python 3 on Windows, Linux, or macOS:

    python reassemble.py

Or on systems where Python 3 is named python3:

    python3 reassemble.py

The original .whl files appear in a new restored directory. The script
checks every part and the final original files with SHA-256 and unpacks
the extra ZIP layer automatically. No third-party Python packages are needed.
To select another destination:

    python reassemble.py --output-dir /path/to/destination

Individual parts cannot be installed or unzipped independently. These are
binary slices, not the .z01 format. The original wheels are Linux x86_64,
CPython 3.12, CUDA 12 / PyTorch 2.10, C++11 ABI enabled builds, as indicated
by their filenames.

These numbered parts are stored as ordinary files in the repository's
wheel-parts directory and are included in a normal Git clone.
