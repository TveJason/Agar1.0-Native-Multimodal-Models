"""Restore the losslessly compressed upstream font after cloning."""
import gzip
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / 'src/proto/assets/cangerjinkai.ttf.gz'
target = source.with_suffix('')
if not target.exists():
    target.write_bytes(gzip.decompress(source.read_bytes()))
print(f'Font ready: {target}')
