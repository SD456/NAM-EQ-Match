import gzip
import hashlib
import json

import pytest
from scripts import materialize_review as materialize


def test_restore_is_lossless_idempotent_and_never_overwrites_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(materialize, 'ROOT', tmp_path.resolve())
    original = b'{"known_result": 0.14808462665377048}\n'
    archive = gzip.compress(original, mtime=0)
    (tmp_path / 'result.json.gz').write_bytes(archive)
    manifest = {'files': [dict(path='result.json', upload_path='result.json.gz', compressed=True,
        upload_sha256=hashlib.sha256(archive).hexdigest(), public_sha256=hashlib.sha256(original).hexdigest(),
        transformed_local_paths=False)]}
    (tmp_path / 'review-snapshot-manifest.json').write_text(json.dumps(manifest))
    materialize.main()
    assert (tmp_path / 'result.json').read_bytes() == original
    materialize.main()
    (tmp_path / 'result.json').write_bytes(b'local changes')
    with pytest.raises(FileExistsError):
        materialize.main()
    assert (tmp_path / 'result.json').read_bytes() == b'local changes'
    (tmp_path / 'result.json.gz').write_bytes(archive + b'corrupt')
    with pytest.raises(ValueError, match='Published bytes changed'):
        materialize.main()


def test_manifest_path_cannot_leave_review_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(materialize, 'ROOT', tmp_path.resolve())
    with pytest.raises(ValueError, match='escapes'):
        materialize.checked_path('../outside.json')
