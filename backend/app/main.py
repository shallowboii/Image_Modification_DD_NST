# backend/app/main.py
import os
import uuid
import json
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.db import init_db, SessionLocal, Job
from app.services.orchestrator import run_cyclic_pipeline, run_adaptive_cyclic_pipeline

app = FastAPI()
init_db()

# ── Absolute directory paths (independent of uvicorn launch directory) ────────
_BACKEND_DIR = Path(__file__).parent.parent
_RESULTS_DIR = _BACKEND_DIR / "results"
_UPLOADS_DIR = _BACKEND_DIR / "uploads"
_RESULTS_DIR.mkdir(exist_ok=True)
_UPLOADS_DIR.mkdir(exist_ok=True)

_MAX_UPLOAD_BYTES   = 20 * 1024 * 1024  # 20 MB
_ALLOWED_TYPES      = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Generate endpoint ─────────────────────────────────────────────────────────

@app.post("/generate")
async def generate(
    file:                            UploadFile = File(...),
    style:                           UploadFile = File(None),
    preset:                          str        = Form(None),
    layer:                           str        = Form(None),
    cycles:                          int        = Form(None, ge=1, le=10),
    num_iterations_nst:              int        = Form(None),
    num_gradient_ascent_iterations:  int        = Form(None),
    pyramid_size:                    int        = Form(None, ge=1, le=10),
    pyramid_ratio:                   float      = Form(None),
    lr:                              float      = Form(None),
    style_weight:                    float      = Form(None),
    mode:                            str        = Form(None),
    background_tasks:                BackgroundTasks = None,
):
    # ── Validate & read content image ─────────────────────────────────────
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content image type '{file.content_type}'. Allowed: {sorted(_ALLOWED_TYPES)}",
        )
    content_bytes = await file.read()
    if len(content_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Content image exceeds {_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.")

    uid        = str(uuid.uuid4())
    input_path = str((_UPLOADS_DIR / f"{uid}_{file.filename}").resolve())
    with open(input_path, "wb") as fh:
        fh.write(content_bytes)

    # ── Validate & read style image (optional) ────────────────────────────
    style_path = None
    if style and style.filename:
        if style.content_type not in _ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported style image type '{style.content_type}'.",
            )
        style_bytes = await style.read()
        if len(style_bytes) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Style image exceeds size limit.")
        style_path = str((_UPLOADS_DIR / f"{uid}_style_{style.filename}").resolve())
        with open(style_path, "wb") as fh:
            fh.write(style_bytes)

    # ── Build the full parameters dict (stored in DB and forwarded to pipeline) ──
    parameters: dict = {}
    if preset is not None:
        parameters["preset"] = preset
    else:
        parameters.update({
            "num_iterations_nst":             num_iterations_nst,
            "pyramid_size":                   pyramid_size,
            "pyramid_ratio":                  pyramid_ratio,
            "num_gradient_ascent_iterations": num_gradient_ascent_iterations,
            "lr":                             lr,
            "cycles":                         cycles,
            "layer":                          layer,
            "style_weight":                   style_weight,
            "preset":                         "Default",
        })
    if style_path:
        parameters["style"] = style_path
    parameters["mode"] = mode

    # ── Create job record ─────────────────────────────────────────────────
    db: Session = SessionLocal()
    try:
        job = Job(
            status="pending",
            preset=preset,
            params=json.dumps(parameters),
            input_path=input_path,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    # ── Background task ───────────────────────────────────────────────────
    def _run(job_id: int, input_path: str, parameters: dict):
        job_results_dir = str(_RESULTS_DIR / str(job_id))
        db = SessionLocal()
        try:
            db.query(Job).filter(Job.id == job_id).update({"status": "running"})
            db.commit()

            print(f"[Job {job_id}] Starting  mode={parameters.get('mode')}  preset={parameters.get('preset')}")

            if parameters.get("mode") == "adaptive":
                out = run_adaptive_cyclic_pipeline(input_path, parameters, dump_dir=job_results_dir)
            else:
                out = run_cyclic_pipeline(input_path, parameters, dump_dir=job_results_dir)

            # Store path relative to _RESULTS_DIR so the frontend can reconstruct the URL
            rel = Path(out).relative_to(_RESULTS_DIR)
            db.query(Job).filter(Job.id == job_id).update({
                "status":      "done",
                "output_path": str(rel).replace("\\", "/"),
            })
            db.commit()
            print(f"[Job {job_id}] Done → {rel}")

        except Exception as e:
            db.query(Job).filter(Job.id == job_id).update({"status": "failed"})
            db.commit()
            print(f"[Job {job_id}] Failed: {e}")
            traceback.print_exc()
        finally:
            db.close()
            # Clean up uploaded files regardless of outcome
            for path in (input_path, parameters.get("style")):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    background_tasks.add_task(_run, job_id, input_path, parameters)
    return {"job_id": job_id}


# ── Job status endpoint ───────────────────────────────────────────────────────

@app.get("/job/{job_id}")
def get_job(job_id: int):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
    finally:
        db.close()
    if not job:
        return {"error": "not found"}
    return {"id": job.id, "status": job.status, "output_path": job.output_path}


# ── Static file serving ───────────────────────────────────────────────────────

app.mount("/results", StaticFiles(directory=str(_RESULTS_DIR)), name="results")

frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "react-app" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "Frontend not built. Run: cd frontend/react-app && npm run build"}
