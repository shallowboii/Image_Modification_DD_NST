# backend/app/main.py
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import os, uuid, json
from app.db import init_db, SessionLocal, Job
from app.services.orchestrator import run_cyclic_pipeline, run_adaptive_cyclic_pipeline
from sqlalchemy.orm import Session
import traceback
from pathlib import Path

app = FastAPI()
init_db()
os.makedirs("results", exist_ok=True)



@app.post("/generate")
async def generate(file: UploadFile = File(...),
                   style: UploadFile = File(None),
                   preset: str = Form(None),
                   layer: str = Form(None),
                   cycles: int = Form(None, ge=1, le=10),
                   num_iterations_nst: int = Form(None),
                   num_gradient_ascent_iterations: int = Form(None),
                   pyramid_size: int = Form(None, ge=1, le=5),
                   pyramid_ratio: float = Form(None),
                   lr: float = Form(None),
                   style_weight: float = Form(None),
                   mode: str = Form(None),
                   background_tasks: BackgroundTasks = None):
    # save content upload
    uid = str(uuid.uuid4())
    upload_dir = Path(__file__).parent.parent / "uploads"
    upload_dir.mkdir(exist_ok=True)

    input_path = str((upload_dir / f"{uid}_{file.filename}").resolve())
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # save style upload if provided
    style_path = None
    if style:
        style_path = str((upload_dir / f"{uid}_style_{style.filename}").resolve())
        with open(style_path, "wb") as f:
            f.write(await style.read())

    # create job
    db: Session = SessionLocal()
    job = Job(status="pending", preset=preset, params=json.dumps({"cycles": cycles}), input_path=input_path)
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    db.close()

    # background run
    def _run(job_id, input_path):
        db = SessionLocal()
        parameters = {}
        db.query(Job).filter(Job.id == job_id).update({"status": "running"})
        db.commit()
        try:
            if preset is not None:
                print("Preset selected:", preset)
                parameters["preset"] = preset
            else:
                print("No preset selected, using individual parameters")
                parameters.update({
                    "num_iterations_nst": num_iterations_nst,
                    "pyramid_size": pyramid_size,
                    "pyramid_ratio": pyramid_ratio,
                    "num_gradient_ascent_iterations": num_gradient_ascent_iterations,
                    "lr": lr,
                    "cycles": cycles,
                    "layer": layer,
                    "style_weight": style_weight,
                    "preset": "Default",
                })
            # Always forward the uploaded style image and pipeline mode,
            # regardless of whether a preset or custom params were used.
            if style_path:
                parameters["style"] = style_path
            parameters["mode"] = mode
            print("Parameters for job:", parameters)
            if mode == "adaptive":
                out = run_adaptive_cyclic_pipeline(input_path, parameters, dump_dir="results")
            else:
                out = run_cyclic_pipeline(input_path, parameters, dump_dir="results")
            db.query(Job).filter(Job.id == job_id).update({"status": "done", "output_path": out})
            db.commit()
        except Exception as e:
            db.query(Job).filter(Job.id == job_id).update({"status": "failed"})
            db.commit()
            print("Job failed", e)
            traceback.print_exc()
            print("Exception type:", type(e))
            print("Exception args:", e.args)
        finally:
            db.close()

    background_tasks.add_task(_run, job_id, input_path)

    return {"job_id": job_id}

@app.get("/job/{job_id}")
def get_job(job_id: int):
    db = SessionLocal()
    job = db.get(Job, job_id)
    db.close()
    if not job:
        return {"error": "not found"}
    return {"id": job.id, "status": job.status, "output_path": job.output_path}

# Serve results dir
app.mount("/results", StaticFiles(directory="results"), name="results")

# Serve the React app as static files
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "react-app" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "Frontend not built. Run: cd frontend/react-app && npm run build"}
