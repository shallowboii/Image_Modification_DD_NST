# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import os, uuid, json
from app.db import init_db, SessionLocal, Job
from app.services.orchestrator import run_cyclic_pipeline
from sqlalchemy.orm import Session
import traceback

app = FastAPI()
init_db()
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

@app.post("/generate")
async def generate(file: UploadFile = File(...),
                   file: UploadFile = File(...),
                   preset: str = Form(None),
                   cycles: int = Form(None),
                   pyramid_size: int = Form(None),
                   pyramid_ratio: float = Form(None),
                   num_gradient_ascent_iterations: int = Form(None),
                   lr: float = Form(None),

                   background_tasks: BackgroundTasks = None):
    # save upload
    uid = str(uuid.uuid4())
    input_path = f"uploads/{uid}_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # create job
    db: Session = SessionLocal()
    job = Job(status="pending", preset=preset, params=json.dumps({"cycles":cycles}), input_path=input_path)
    db.add(job); db.commit(); db.refresh(job)

    # background run
    def _run(job_id, input_path):
        db = SessionLocal()
        job = db.query(Job).get(job_id)
        db.query(Job).filter(Job.id==job_id).update({"status":"running"}); db.commit()
        try:
            overrides = {}
            if pyramid_size is not None: overrides["pyramid_size"]=pyramid_size
            if pyramid_ratio is not None: overrides["pyramid_ratio"]=pyramid_ratio
            if num_gradient_ascent_iterations is not None: overrides["num_gradient_ascent_iterations"]=num_gradient_ascent_iterations
            if lr is not None: overrides["lr"]=lr
            overrides["cycles"]=cycles
            out = run_cyclic_pipeline(input_path, preset, overrides, dump_dir="results")
            db.query(Job).filter(Job.id==job_id).update({"status":"done","output_path":out}); db.commit()
        except Exception as e:
            db.query(Job).filter(Job.id==job_id).update({"status":"failed"}); db.commit()
            print("Job failed", e)
            traceback.print_exc() 
            print("Exception type:", type(e))
            print("Exception args:", e.args)

    background_tasks.add_task(_run, job.id, input_path)

    return {"job_id": job.id}

@app.get("/job/{job_id}")
def get_job(job_id: int):
    db = SessionLocal()
    job = db.query(Job).get(job_id)
    if not job:
        return {"error":"not found"}
    return {"id": job.id, "status": job.status, "output_path": job.output_path}

# Serve results dir
app.mount("/results", StaticFiles(directory="results"), name="results")