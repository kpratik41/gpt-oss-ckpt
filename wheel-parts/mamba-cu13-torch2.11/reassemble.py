"""Reassemble and verify the original wheels using Python 3 (no dependencies)."""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Destination (default: restored beside this script)')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    destination = args.output_dir or base / 'restored'
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        target = destination / entry['original_name']
        if target.exists():
            if target.stat().st_size == entry['original_bytes'] and digest(target) == entry['original_sha256']:
                print('Already verified:', target)
                continue
            raise SystemExit('Refusing to overwrite an existing different file: ' + str(target))
        with tempfile.TemporaryDirectory(prefix='wheel-join-', dir=destination) as temporary:
            temp = Path(temporary)
            payload = temp / entry['payload_name']
            payload_hash = hashlib.sha256()
            with payload.open('wb') as output:
                for item in entry['parts']:
                    part = base / item['name']
                    if not part.is_file():
                        raise SystemExit('Missing part: ' + str(part))
                    part_hash = hashlib.sha256()
                    size = 0
                    with part.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            output.write(block)
                            part_hash.update(block)
                            payload_hash.update(block)
                            size += len(block)
                    if size != item['bytes'] or part_hash.hexdigest() != item['sha256']:
                        raise SystemExit('Corrupt part: ' + str(part))
            if payload.stat().st_size != entry['payload_bytes'] or payload_hash.hexdigest() != entry['payload_sha256']:
                raise SystemExit('Payload verification failed: ' + entry['payload_name'])
            restored = temp / entry['original_name']
            if entry['compression'] == 'zip':
                with zipfile.ZipFile(payload) as archive:
                    with archive.open(entry['original_name']) as source, restored.open('wb') as output:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            output.write(block)
            elif entry['compression'] != 'none':
                raise SystemExit('Unknown compression: ' + entry['compression'])
            if restored.stat().st_size != entry['original_bytes'] or digest(restored) != entry['original_sha256']:
                raise SystemExit('Original wheel verification failed: ' + entry['original_name'])
            restored.rename(target)
            print('Restored and SHA-256 verified:', target)


if __name__ == '__main__':
    main()
