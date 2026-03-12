# backend/app/main.py
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import os, uuid, json
from app.db import init_db, SessionLocal, Job
from app.services.orchestrator import run_cyclic_pipeline
from sqlalchemy.orm import Session
import traceback
from pathlib import Path

app = FastAPI()
init_db()
os.makedirs("uploads", exist_ok=True)   
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
        parameters = {}
        job = db.query(Job).get(job_id)
        db.query(Job).filter(Job.id==job_id).update({"status":"running"}); db.commit()
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
                "preset": "Default",
                "style": style.filename if style else None,
            })
            print (layer)
            print(parameters.get("layer"))
            print("Parameters for job:", parameters)
            out = run_cyclic_pipeline(input_path, parameters, dump_dir="results")
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

# Serve the React app as static files
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "react-app" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "Frontend not built. Run: cd frontend/react-app && npm run build"}