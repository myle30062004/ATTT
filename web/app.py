"""Web demo for the ML-based NIDS.

    python -m uvicorn web.app:app --host 127.0.0.1 --port 8000        # from network-ids/
    NIDS_ALLOW_LIVE=1 sudo -E python -m uvicorn web.app:app ...         # Kali: enable live capture

Pages (single-page UI in web/static): overview, PCAP analysis, CSV flow analysis,
single-flow explanation, realtime monitor (replay / live), model results.
"""
from __future__ import annotations

import asyncio
import queue
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from web import services as svc  # noqa: E402
from web.realtime import LIVE_ENABLED, CaptureError, Session, interfaces  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = Path(tempfile.gettempdir()) / "nids_web_uploads"
UPLOADS.mkdir(exist_ok=True)
MAX_UPLOAD_MB = 300

app = FastAPI(title="NIDS – Web demo", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/figures", StaticFiles(directory=ROOT / "reports" / "figures"), name="figures")


@app.on_event("startup")
def _warm() -> None:
    svc.pipeline()          # load bundle once (LightGBM + GMM + cleaner)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {**svc.model_info(), "live_enabled": LIVE_ENABLED, "interfaces": interfaces()}


@app.get("/api/samples")
def samples():
    return svc.list_samples()


@app.get("/api/results")
def results():
    return svc.model_results()


# ------------------------------------------------------------------ analysis jobs
def _save_upload(file: UploadFile, allowed: tuple[str, ...]) -> Path:
    name = Path(file.filename or "upload").name
    if not name.lower().endswith(allowed):
        raise HTTPException(400, f"Chỉ nhận file {', '.join(allowed)}")
    dest = UPLOADS / f"{svc.uuid.uuid4().hex[:8]}_{name}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh, length=1 << 20)
    if dest.stat().st_size > MAX_UPLOAD_MB * 2**20:
        dest.unlink()
        raise HTTPException(413, f"File lớn hơn {MAX_UPLOAD_MB} MB")
    return dest


@app.post("/api/analyze/upload")
def analyze_upload(file: UploadFile = File(...)):
    path = _save_upload(file, (".pcap", ".pcapng", ".cap", ".csv"))
    source = {"name": file.filename, "kind": "csv" if path.suffix.lower() == ".csv" else "pcap", "uploaded": True}
    fn = svc.analyze_csv if source["kind"] == "csv" else svc.analyze_pcap
    return svc.start_job(source["kind"], file.filename, fn, path, source).to_dict(False)


@app.post("/api/analyze/sample/{sample_id}")
def analyze_sample(sample_id: str):
    try:
        path, meta = svc.sample_path(sample_id)
    except KeyError:
        raise HTTPException(404, "Không có mẫu này")
    fn = svc.analyze_csv if meta["kind"] == "csv" else svc.analyze_pcap
    return svc.start_job(meta["kind"], meta["title"], fn, path, meta, cache_key=f"sample_{sample_id}").to_dict(False)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = svc.JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy job")
    return job.to_dict()


# ------------------------------------------------------------------ single flow
class FlowIn(BaseModel):
    features: dict[str, float | None]


@app.get("/api/flow/random")
def flow_random(family: str = "any"):
    return svc.random_flow(family)


@app.post("/api/flow/predict")
def flow_predict(body: FlowIn):
    return svc.predict_flow(body.features)


# ------------------------------------------------------------------ realtime
@app.websocket("/ws/realtime")
async def ws_realtime(ws: WebSocket):
    await ws.accept()
    session = None
    try:
        params = await ws.receive_json()
        if params.get("mode") == "live":
            session = Session("live", interface=params.get("interface") or None)
        else:
            path, _ = svc.sample_path(params.get("sample", ""))
            session = Session("replay", str(path), speed=float(params.get("speed", 50)),
                              max_packets=params.get("max_packets"))
        session.start()
        await ws.send_json({"type": "started", "mode": session.mode})
        while True:
            try:
                msg = session.updates.get_nowait()
            except queue.Empty:
                try:   # browser may send {"action":"stop"}
                    cmd = await asyncio.wait_for(ws.receive_json(), timeout=0.25)
                    if cmd.get("action") == "stop":
                        session.stop()
                except asyncio.TimeoutError:
                    pass
                continue
            await ws.send_json(msg)
            if msg["type"] in ("done", "error"):
                break
    except (CaptureError, KeyError) as exc:
        await ws.send_json({"type": "error", "message": str(exc) or "Không tìm thấy mẫu"})
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            session.stop()
        try:
            await ws.close()
        except RuntimeError:
            pass
