"""Acquire author-linked weights into the shared model store, never into Git.

Uses temporary files, size/hash/ZIP CRC checks, exclusive publication and receipts.
No model code, inference, training or unsafe checkpoint unpickling is executed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
from pathlib import Path
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

OTIS_REVISION = '30282e9e2fd98511f9e33d90c2b9d86d5ae874e0'
ECHO_REVISION = '03874a5f203695f38968068c21584656d475b6b1'
OTIS_ID = '1AZVqME8nX5V2RjvF3moBPregb1lUDnxP'
ECHO_URL = 'https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip'
ECHO_SHA = 'b29362b6b40e8695b138d191f874ad92417b6da9fcdb2f0764ed1440be5272b3'


class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and a.get('id') == 'download-form':
            self.action = a.get('action')
        if tag == 'input' and a.get('name'):
            self.fields[a['name']] = a.get('value', '')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url, path, size, expected_sha=None):
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    started = time.monotonic()
    for attempt in range(3):
        response = opener.open(urllib.request.Request(url, headers={'User-Agent': 'NOAH-weight-acquisition/1'}), timeout=45)
        if 'text/html' not in response.headers.get('Content-Type', ''):
            break
        page = response.read(2_000_000).decode(); response.close()
        form = Confirmation(); form.feed(page)
        if not form.action:
            raise RuntimeError('Download returned HTML without a public download confirmation form')
        if urllib.parse.urlparse(form.action).hostname != 'drive.usercontent.google.com':
            raise RuntimeError('Unexpected download confirmation destination')
        url = form.action + '?' + urllib.parse.urlencode(form.fields)
    else:
        raise RuntimeError('Public download confirmation loop')
    count = 0; last_print = time.monotonic()
    with response, path.open('xb') as out:
        while block := response.read(2**20):
            if time.monotonic() - started > 900:
                raise RuntimeError('15-minute per-asset download cap')
            count += len(block)
            if count > size:
                raise RuntimeError('Download exceeds declared release size')
            out.write(block)
            if time.monotonic() - last_print > 20:
                print(json.dumps({'download': path.name, 'bytes': count, 'expected': size}), flush=True)
                last_print = time.monotonic()
    if count != size:
        raise RuntimeError(f'Incomplete asset: {count} != {size}')
    digest = sha(path)
    if expected_sha and digest != expected_sha:
        raise RuntimeError('Release checksum mismatch')
    return digest


def publish(source, target):
    if target.exists():
        if sha(source) != sha(target):
            raise RuntimeError('Existing model weights differ; refusing overwrite')
        return
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, 2**20)
    if sha(source) != sha(target):
        raise RuntimeError('Model publication checksum mismatch')
    target.chmod(0o444)


def acquire(root, kind):
    if shutil.disk_usage(root).free < 3 * 2**30:
        raise RuntimeError('At least3GiB free space required for bounded acquisition')
    directory = root / ('otis/author-base-2024' if kind == 'otis' else 'echoprime/v1.0.0')
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'asset_manifest.json'
    if receipt.exists():
        old = json.loads(receipt.read_text()); weight = directory / old['filename']
        if sha(weight) != old['weight_sha256']:
            raise RuntimeError('Existing registered weight failed checksum verification')
        print(json.dumps({'asset': kind, 'status': 'EXISTING_HASH_VERIFIED', 'path': str(weight)}), flush=True)
        return
    with tempfile.TemporaryDirectory(prefix='.acquire-', dir=directory) as temporary:
        temp = Path(temporary)
        if kind == 'otis':
            url = 'https://drive.usercontent.google.com/download?' + urllib.parse.urlencode(dict(id=OTIS_ID, export='download', confirm='t'))
            archive = temp / 'otis_base.pth'
            digest = download(url, archive, 108005017)
            target = directory / 'otis_base.pth'; publish(archive, target)
            record = dict(author_repository='https://github.com/oetu/otis', repository_revision=OTIS_REVISION,
                author_folder='https://drive.google.com/drive/folders/1sMxJwvyZY7M2Z_gykcjLgIXa13iYDCEf',
                source_file_id=OTIS_ID, source_url=url, upstream_checksum=None,
                checksum_assurance='local acquisition SHA256; upstream did not publish a checksum')
        else:
            archive = temp / 'model_data.zip'; digest = download(ECHO_URL, archive, 1338009590, ECHO_SHA)
            with zipfile.ZipFile(archive) as z:
                members = [i for i in z.infolist() if i.filename == 'model_data/weights/echo_prime_encoder.pt']
                if len(members) != 1 or members[0].file_size > 512 * 2**20:
                    raise RuntimeError('Missing/duplicate/oversized video encoder member')
                member = members[0]; staged = temp / 'echo_prime_encoder.pt'
                with z.open(member) as src, staged.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 2**20)
                target = directory / staged.name; publish(staged, target)
            record = dict(author_repository='https://github.com/echonet/EchoPrime', repository_revision=ECHO_REVISION,
                release_tag='v1.0.0', source_url=ECHO_URL, source_archive_sha256=digest,
                upstream_checksum=ECHO_SHA, archive_member=member.filename, member_crc32=f'{member.CRC:08x}',
                checksum_assurance='GitHub published archive SHA256 verified; extracted member CRC checked',
                retained_scope='video encoder only; no text/view/retrieval weights or report candidates')
        record.update(schema_version=1, asset=kind, acquired_at_utc=datetime.now(timezone.utc).isoformat(),
            filename=target.name, bytes=target.stat().st_size, weight_sha256=sha(target),
            status='DOWNLOADED_NOT_YET_PIPELINE_VERIFIED', noah_checkpoint_identity='public upstream candidate, not established NOAH author checkpoint')
        with receipt.open('x') as f:
            json.dump(record, f, indent=2); f.write('\n')
        receipt.chmod(0o444)
        print(json.dumps({'asset': kind, **record, 'path': str(target)}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/n/data1/hms/dbmi/rajpurkar/lab/home/wey371/models'))
    p.add_argument('--asset', choices=['otis', 'echoprime', 'all'], default='all')
    args = p.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    for kind in (['otis', 'echoprime'] if args.asset == 'all' else [args.asset]):
        acquire(args.root, kind)


if __name__ == '__main__':
    main()
