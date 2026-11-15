import sys
import os
from huggingface_hub import HfApi
# Compares every mirrored file with its source (LFS SHA-256, or git blob id for small files).
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mirror.py')).read().split("name = sys.argv[1]")[0]
ns = {}; exec(src, ns)
api = HfApi()
revs = {'lfm3': '1ef64c82a7e6d3187a74f9103f70d733ddb9ed52', 'qwen': 'b63dbd33b9d3b9b96b711f060e4dddfe66993ef5', 'gemma': '9847447e160486f74b4fb4a90472b5da0c5bf647'}
def key(i):
    return i.lfs.sha256 if getattr(i, 'lfs', None) else i.blob_id
bad = 0
for name, s in ns['SPECS'].items():
    dst = {i.path: key(i) for i in api.get_paths_info(s['repo'], [f['dest'] for f in s['files']], revision=revs[name])}
    for f in s['files']:
        o = api.get_paths_info(f['repo'], [f['path']], revision=f['rev'])[0]
        ok = key(o) == dst.get(f['dest'])
        bad += not ok
        if not ok: print('DIFF', name, f['dest'])
    print(name, len(s['files']), 'files checked')
print('mismatches', bad)
