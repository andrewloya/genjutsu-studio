#!/usr/bin/env python3
"""Genjutsu Studio: a local LYRC-style page for gj.py.

Pick a clip, trim it, pick a character, set lip sync, press Make. The server runs the free prep on this Mac,
sends ONE Evolink render, collects it, and lists every swap with its lip-sync score and real cost.
One job at a time, by design: spend as little as possible.

  .venv/bin/python studio.py            -> http://127.0.0.1:8790
"""
import json, os, re, shutil, subprocess, sys, threading, time, uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / '.venv/bin/python')
GJ = str(ROOT / 'gj.py')
JOBS, SOURCES, CHARS = ROOT / 'jobs', ROOT / 'sources', ROOT / 'characters'
for d in (JOBS, SOURCES, CHARS, SOURCES / '.thumbs', SOURCES / '.proxies'): d.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import gj  # noqa: E402  (api helper + probe only; heavy work runs in subprocesses)
gj.env()

STYLE = {'max': 'wire', 'balanced': 'lips', 'clean': 'features'}
STYLE_BACK = {v: k for k, v in STYLE.items()}
PER_S_DRAFT = 2 * 5.71 * 0.01471   # 480p video-edit bills input + output seconds; 8 s -> $1.34 (measured $1.326)
PER_S_HD = 63.3 * 0.01471          # draft-to-video 1080p (~950 credits for 15 s)
PREP_STEPS = ['Cutting your clip', 'Pulling your vocals', 'Masking you out', 'Building the heat-map body', 'Drawing the face guide', 'Packing it for Seedance']

app = FastAPI()
lock = threading.Lock()
task = {}  # the one active task: {id, kind, job, run, stage, step, steps, progress, error, started}


def slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:40] or 'x'


def rel(p):
    return '/files/' + str(Path(p).resolve().relative_to(ROOT))


def proxy(src):
    """Light 540p silent copy for the trim preview (the real clip can be a 200 MB 4K file)."""
    out = SOURCES / '.proxies' / (src.stem + '.mp4')
    if not out.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-vf', "scale='if(gt(iw,ih),-2,540)':'if(gt(iw,ih),540,-2)'", '-an',
                        '-c:v', 'libx264', '-crf', '27', '-preset', 'veryfast', '-g', '12', '-movflags', '+faststart', str(out)])
    return out


def thumb(video, out, t=1.0):
    if not out.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', str(video), '-frames:v', '1', '-vf', 'scale=-2:360', '-q:v', '4', str(out)])
    return out


def today_spend():
    day = time.strftime('%Y-%m-%d'); total = 0.0
    for led in list(JOBS.glob('*/ledger.jsonl')) + [CHARS / 'ledger.jsonl']:
        if not led.exists(): continue
        for line in led.read_text().splitlines():
            try: j = json.loads(line)
            except ValueError: continue
            if j.get('t', '').startswith(day): total += ((j.get('usage') or {}).get('cost') or {}).get('usd', 0) or 0
    return round(total, 2)


# ---------------------------------------------------------------- state

def characters():
    out = []
    for d in sorted(CHARS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir(): continue
        img = next((p for p in d.glob('ref.*')), None)
        if not img: continue
        meta = json.loads((d / 'meta.json').read_text()) if (d / 'meta.json').exists() else {}
        out.append({'id': d.name, 'name': meta.get('name', d.name), 'description': meta.get('description', ''), 'image': rel(img)})
    return out


_probe = {}


def sources():
    out = []
    for p in sorted(SOURCES.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.suffix.lower() not in ('.mp4', '.mov', '.m4v'): continue
        key = (p.name, p.stat().st_mtime)
        try: w, h, d, a = _probe[key] if key in _probe else _probe.setdefault(key, gj.probe(p))
        except Exception: continue
        out.append({'id': p.name, 'name': p.stem, 'url': rel(proxy(p)), 'thumb': rel(thumb(p, SOURCES / '.thumbs' / (p.stem + '.jpg'))),
                    'duration': round(d, 2), 'w': w, 'h': h, 'audio': a})
    return out


def run_list():
    runs = []
    for job in JOBS.iterdir():
        man_p = job / 'manifest.json'
        if not man_p.exists(): continue
        man = json.loads(man_p.read_text()); meta = json.loads((job / 'runs-meta.json').read_text()) if (job / 'runs-meta.json').exists() else {}
        for rp in (job / 'runs').glob('*.json'):
            name = rp.stem
            if name.endswith('-sync'): continue
            j = json.loads(rp.read_text()); fin = j.get('final') or {}; m = meta.get(name.removesuffix('-hd'), {})
            sync_p = job / 'runs' / f'{name}-sync.json'; sync = json.loads(sync_p.read_text()) if sync_p.exists() else {}
            final = job / 'runs' / f'{name}-final.mp4'
            status = fin.get('status') or ('rendering' if not j.get('done') else 'unknown')
            if status == 'completed' and not final.exists(): status = 'finishing'
            item = {'id': f'{job.name}/{name}', 'job': job.name, 'run': name, 'hd': name.endswith('-hd'), 'status': status,
                    'error': (fin.get('error') or {}).get('message') if isinstance(fin.get('error'), dict) else fin.get('error'),
                    'character': m.get('character', 'Blue glass'), 'lipsync': m.get('lipsync') or STYLE_BACK.get(man.get('mesh_style', 'wire'), 'max'),
                    'cost': ((fin.get('usage') or {}).get('cost') or {}).get('usd'), 'lip': sync.get('mouth_corr'),
                    'outfit': 'head only' if m.get('swap') == 'head' else (m.get('sleeves') + ' sleeves') if m.get('match_outfit') and m.get('sleeves') else None,
                    'clip': Path(man.get('source', '')).stem, 'start': man.get('start'), 'length': man.get('length'),
                    'created': rp.stat().st_mtime, 'final': rel(final) if final.exists() else None}
            if final.exists():
                cmp = job / 'runs' / f'{name}-compare.mp4'
                item.update(compare=rel(cmp) if cmp.exists() else None, guide=rel(job / 'input.mp4'), original=rel(job / 'original.mp4'),
                            thumb=rel(thumb(final, job / 'runs' / f'{name}-thumb.jpg', 1.6)),
                            hd_ok=not item['hd'] and time.time() - rp.stat().st_mtime < 23 * 3600 and not (job / 'runs' / f'{name}-hd.json').exists(),
                            hd_cost=round(man.get('length', 8) * PER_S_HD, 2))
            runs.append(item)
    return sorted(runs, key=lambda r: r['created'], reverse=True)


@app.get('/api/state')
def state():
    with lock: t = dict(task)
    return {'characters': characters(), 'sources': sources(), 'runs': run_list(), 'spend_today': today_spend(), 'task': t or None,
            'page_version': int((ROOT / 'static' / 'studio.html').stat().st_mtime), 'has_key': bool(os.environ.get('EVOLINK_API_KEY')),
            'rates': {'draft_per_s': PER_S_DRAFT, 'hd_per_s': PER_S_HD}}


# ---------------------------------------------------------------- uploads

@app.post('/api/sources')
async def add_source(file: UploadFile = File(...)):
    name = slug(Path(file.filename).stem) + Path(file.filename).suffix.lower()
    dest = SOURCES / name
    with open(dest, 'wb') as f: shutil.copyfileobj(file.file, f)
    try: w, h, d, a = gj.probe(dest)
    except Exception: dest.unlink(); raise HTTPException(400, "That file isn't a video we can read.")
    if not a: dest.unlink(); raise HTTPException(400, 'This clip has no sound. Lip sync needs your vocals in the clip.')
    if d < 4: dest.unlink(); raise HTTPException(400, 'This clip is under 4 seconds. Seedance needs at least 4.')
    proxy(dest)
    return {'id': name}


@app.post('/api/characters')
async def add_character(file: UploadFile = File(...), name: str = Form(...), description: str = Form('')):
    d = CHARS / slug(name); d.mkdir(exist_ok=True)
    for old in d.glob('ref.*'): old.unlink()
    with open(d / ('ref' + (Path(file.filename).suffix.lower() or '.png')), 'wb') as f: shutil.copyfileobj(file.file, f)
    (d / 'meta.json').write_text(json.dumps({'name': name, 'description': description, 'made': time.strftime('%Y-%m-%d %H:%M:%S')}))
    return {'id': d.name}


@app.post('/api/characters/generate')
def generate_character(body: dict):
    name, desc = body.get('name', '').strip(), body.get('description', '').strip()
    if not name or not desc: raise HTTPException(400, 'Give the character a name and a description.')
    start('character', lambda t: _gen_character(t, name, desc), job=None)
    return {'ok': True}


def _gen_character(t, name, desc):
    prompt = (f'Full-body character reference, front view, standing straight with arms relaxed, the whole body visible from head to shoes, centered, '
              f'plain light-gray studio background, soft even light. {desc} Clearly defined lips, a visible mouth and expressive eyes. '
              'High detail, clean render, no text, no logos.')
    t.update(stage='Making your character', steps=['Making your character'], step=0)
    r = json.loads(gj.api('POST', '/images/generations', {'model': 'gemini-nano-banana-2.1', 'prompt': prompt, 'size': '9:16', 'quality': '2K'}))
    tid = r.get('id')
    if not tid: raise RuntimeError('The image model said no: ' + json.dumps(r.get('error') or r)[:200])
    for _ in range(90):
        s = json.loads(gj.api('GET', '/tasks/' + tid) or '{}')
        if s.get('status') == 'completed':
            d = CHARS / slug(name); d.mkdir(exist_ok=True)
            u = s['results'][0]; ext = Path(u.split('?')[0]).suffix or '.png'
            for old in d.glob('ref.*'): old.unlink()
            subprocess.run(['curl', '-sfL', '-A', 'Mozilla/5.0', '-o', str(d / f'ref{ext}'), u], check=True)
            (d / 'meta.json').write_text(json.dumps({'name': name, 'description': desc, 'made': time.strftime('%Y-%m-%d %H:%M:%S')}))
            with open(CHARS / 'ledger.jsonl', 'a') as f: f.write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'character': name, 'usage': s.get('usage')}) + '\n')
            return
        if s.get('status') == 'failed': raise RuntimeError('The image model failed: ' + json.dumps(s.get('error'))[:200])
        t['progress'] = s.get('progress'); time.sleep(4)
    raise RuntimeError('The character is taking too long. Try again in a minute.')


# ---------------------------------------------------------------- make / hd

def start(kind, fn, job):
    with lock:
        if task and not task.get('done'):
            raise HTTPException(409, 'One thing at a time: wait for the current one to finish.')
        task.clear(); task.update(id=uuid.uuid4().hex[:8], kind=kind, job=job, stage='Starting', step=0, steps=[], progress=None, error=None, done=False, started=time.time())
        t = task

    def wrap():
        try: fn(t)
        except Exception as e: t['error'] = str(e)[:300]
        finally: t['done'] = True; t['ended'] = time.time()
    threading.Thread(target=wrap, daemon=True).start()


@app.post('/api/make')
def make(body: dict):
    src = SOURCES / body['source']
    if not src.exists(): raise HTTPException(400, 'Pick a clip first.')
    char = CHARS / body['character']
    if not char.exists(): raise HTTPException(400, 'Pick a character first.')
    lip = body.get('lipsync', 'clean'); style = STYLE.get(lip, 'wire')
    s0, ln = round(float(body.get('start', 0)), 2), round(float(body.get('length', 8)), 2)
    if not 4 <= ln <= 15: raise HTTPException(400, 'Length has to be 4 to 15 seconds.')
    swap = 'head' if body.get('swap') == 'head' else 'body'
    job = JOBS / f'{slug(src.stem)}-{s0:g}-{ln:g}-{style}{"-head" if swap == "head" else ""}-p{gj.PREP_VERSION}'
    match = bool(body.get('match_outfit', True)); relaxed = bool(body.get('relaxed'))
    pitch = int(body.get('pitch', 3))  # 0 for other artists' songs: send their vocals as they are
    skip = str(body.get('skip', ''))  # clip-second ranges left untouched (shots of other people)
    if skip: job = job.with_name(job.name + '-skip' + skip.replace('.', '_').replace(',', '+'))
    if pitch != 3: job = job.with_name(job.name + f'-pitch{pitch}')
    meta = json.loads((char / 'meta.json').read_text()) if (char / 'meta.json').exists() else {}
    desc = (body.get('description') or meta.get('description') or meta.get('name') or 'the character in the reference image').strip()
    dry = bool(body.get('dry'))  # prep only, no render: free, for testing the guide
    start('make', lambda t: _make(t, src, job, s0, ln, style, lip, char, desc, dry, match, relaxed, pitch, skip), job=job.name)
    return {'ok': True}


def _make(t, src, job, s0, ln, style, lip, char, desc, dry=False, match=True, relaxed=False, pitch=3, skip=''):
    job.mkdir(exist_ok=True)
    t['steps'] = PREP_STEPS + ['Rendering on Evolink', 'Putting your song back']
    if not (job / 'manifest.json').exists():
        log = open(job / 'prep.log', 'w')
        p = subprocess.Popen([PY, '-u', GJ, 'prep', str(src), str(job), '--start', str(s0), '--len', str(ln), '--mesh-style', style,
                              '--swap', 'head' if '-head-' in job.name else 'body', '--pitch', str(pitch)] + (['--skip', skip] if skip else []),
                             stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        while p.poll() is None:
            m = re.findall(r'^\[(\d)/6\]', (job / 'prep.log').read_text(errors='ignore'), re.M)
            if m: t.update(step=int(m[-1]) - 1, stage=PREP_STEPS[int(m[-1]) - 1])
            prog = re.findall(r'(?:depth|deeplab) (\d+)/(\d+)', (job / 'prep.log').read_text(errors='ignore'))
            t['progress'] = round(100 * int(prog[-1][0]) / int(prog[-1][1])) if prog else None
            time.sleep(1.5)
        if p.returncode or not (job / 'manifest.json').exists():
            tail = [l for l in (job / 'prep.log').read_text(errors='ignore').splitlines() if l.strip() and 'objc' not in l][-3:]
            shutil.rmtree(job, ignore_errors=True)
            raise RuntimeError('Prep failed: ' + ' | '.join(tail)[:250])
    if dry: t.update(step=6, stage='Prepped (no render)', progress=None); return
    t.update(step=6, stage='Rendering on Evolink', progress=None)
    runs = job / 'runs'; runs.mkdir(exist_ok=True)
    n = 1 + len([p for p in runs.glob(f'{char.name}*_a.json')])
    prefix = f'{char.name}{n}_'; name = prefix + 'a'
    ref = next(char.glob('ref.*'))
    r = subprocess.run([PY, GJ, 'submit', str(job), '--ref', str(ref), '--name', prefix, '--character', desc] + ([] if match else ['--no-match-outfit']) + (['--relaxed'] if relaxed else []),
                       capture_output=True, text=True, cwd=ROOT)
    if not (runs / f'{name}.json').exists(): raise RuntimeError('Evolink did not take it: ' + (r.stdout + r.stderr)[-250:])
    resp = json.loads(json.loads((runs / f'{name}.json').read_text())['resp'] or '{}')
    if not resp.get('id'): raise RuntimeError('Evolink said no: ' + json.dumps(resp.get('error') or resp)[:250])
    meta_p = job / 'runs-meta.json'; meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta[name] = {'character': json.loads((char / 'meta.json').read_text()).get('name', char.name) if (char / 'meta.json').exists() else char.name,
                  'character_id': char.name, 'lipsync': lip, 'description': desc, 'match_outfit': match,
                  'swap': json.loads((job / 'manifest.json').read_text()).get('swap', 'body'),
                  'sleeves': (json.loads((job / 'manifest.json').read_text()).get('outfit') or {}).get('sleeves')}
    meta_p.write_text(json.dumps(meta, indent=2))
    t['run'] = f'{job.name}/{name}'
    collect(t, job, name, first_step=6)


def collect(t, job, name, first_step):
    """Poll Evolink for this run; when it lands, gj.py fetch downloads it, conforms it and puts the original audio back."""
    for _ in range(200):
        s = json.loads(gj.api('GET', '/tasks/' + json.loads(json.loads((job / 'runs' / f'{name}.json').read_text())['resp'])['id']) or '{}')
        st = s.get('status'); t['progress'] = s.get('progress')
        if st in ('completed', 'failed'):
            if st == 'completed': t.update(step=first_step + 1, stage='Putting your song back', progress=None)
            subprocess.run([PY, GJ, 'fetch', str(job)], capture_output=True, text=True, cwd=ROOT)
            if st == 'failed': raise RuntimeError('Evolink failed it (not charged): ' + json.dumps(s.get('error'))[:200])
            return
        time.sleep(10)
    raise RuntimeError('Still rendering after 30 min. It will show up here once Evolink finishes.')


@app.post('/api/hd')
def hd(body: dict):
    job, run = JOBS / body['job'], body['run']
    if not (job / 'runs' / f'{run}.json').exists(): raise HTTPException(404, 'No such swap.')

    def go(t):
        t.update(steps=['Upscaling to 1080p on Evolink', 'Putting your song back'], step=0, stage='Upscaling to 1080p on Evolink')
        r = subprocess.run([PY, GJ, 'hd', str(job), run], capture_output=True, text=True, cwd=ROOT)
        name = f'{run}-hd'
        if not (job / 'runs' / f'{name}.json').exists(): raise RuntimeError('Evolink did not take it: ' + (r.stdout + r.stderr)[-200:])
        resp = json.loads(json.loads((job / 'runs' / f'{name}.json').read_text())['resp'] or '{}')
        if not resp.get('id'): raise RuntimeError('Evolink said no: ' + json.dumps(resp.get('error') or resp)[:250])
        t['run'] = f'{job.name}/{name}'
        collect(t, job, name, first_step=0)
    start('hd', go, job=job.name)
    return {'ok': True}


@app.post('/api/task/clear')
def clear_task():
    with lock:
        if task.get('done'): task.clear()
    return {'ok': True}


def key_ok(key):
    r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'https://api.evolink.ai/v1/videos/models', '-H', 'Authorization: Bearer ' + key],
                       capture_output=True, text=True)
    return r.stdout.strip() == '200'


@app.post('/api/key')
def save_key(body: dict):
    """Store the user's own Evolink key in this folder's .env (never sent anywhere but Evolink)."""
    key = (body.get('key') or '').strip()
    if not key or not key_ok(key): raise HTTPException(400, "Evolink didn't accept that key. Copy it again from evolink.ai (API Keys).")
    env_p = ROOT / '.env'; lines = [l for l in (env_p.read_text().splitlines() if env_p.exists() else []) if not l.startswith('EVOLINK_API_KEY=')]
    env_p.write_text('\n'.join(lines + [f'EVOLINK_API_KEY={key}']) + '\n'); os.chmod(env_p, 0o600)
    os.environ['EVOLINK_API_KEY'] = key
    return {'ok': True}


@app.post('/api/reveal')
def reveal(body: dict):
    p = (ROOT / body['path'].removeprefix('/files/')).resolve()
    if ROOT not in p.parents: raise HTTPException(400, 'bad path')
    if sys.platform == 'darwin': subprocess.run(['open', '-R', str(p)])
    elif shutil.which('explorer.exe'): subprocess.run(['explorer.exe', '/select,', subprocess.run(['wslpath', '-w', str(p)], capture_output=True, text=True).stdout.strip()])
    else: subprocess.run(['xdg-open', str(p.parent)])
    return {'ok': True}


def resume():
    """After a restart, keep collecting any render Evolink still has in flight (never resubmits)."""
    for rp in JOBS.glob('*/runs/*.json'):
        if rp.stem.endswith('-sync'): continue
        j = json.loads(rp.read_text())
        if not j.get('done') and 'resp' in j:
            job, name = rp.parent.parent, rp.stem
            try: start('make', lambda t, job=job, name=name: (t.update(steps=['Rendering on Evolink', 'Putting your song back'], stage='Rendering on Evolink', run=f'{job.name}/{name}'),
                                                              collect(t, job, name, first_step=0)), job=job.name)
            except HTTPException: pass
            return


for _d in (JOBS, SOURCES, CHARS): app.mount(f'/files/{_d.name}', StaticFiles(directory=_d), name=_d.name)  # media only, not the code


@app.get('/')
def index():
    return FileResponse(ROOT / 'static' / 'studio.html', headers={'Cache-Control': 'no-store'})


if __name__ == '__main__':
    import uvicorn
    resume()
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('PORT', 8790)), log_level='warning')
