"""
Background job utilities for ModelForge.

Responsibilities:
- Enqueue training jobs
- Track job status & progress
- Provide job metadata for UI polling

Uses:
- Redis
- RQ (Redis Queue)
"""

from typing import Dict, Optional, List
import redis
from rq import Queue
from rq.job import Job
from uuid import uuid4
from datetime import datetime
import json

# =====================================================
# REDIS CONFIG
# =====================================================

redis_conn = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    health_check_interval=30,
)

training_queue = Queue(
    name="training",
    connection=redis_conn,
    default_timeout=-1,  # NO TIMEOUT
)

# =====================================================
# INTERNAL HELPERS
# =====================================================

def _job_key(job_id: str) -> str:
    return f"job:{job_id}"

# TTL: keep job status in Redis for 1 hour
JOB_TTL_SECONDS = 3600


# =====================================================
# ENQUEUE TRAINING JOB (STRING IMPORT — NO CIRCULAR)
# =====================================================

def enqueue_training_job(
    *,
    project_id: int,
    dataset_id: int,
    algorithm: str,
    problem_type: str,
    prep_config: Dict,
    user_params: Dict,
) -> str:

    job_id = str(uuid4())

    # Initial job state with TTL
    redis_conn.hset(
        _job_key(job_id),
        mapping={
            "status": "queued",
            "progress": 0,
            "message": "Job queued",
            "project_id": project_id,
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "created_at": datetime.utcnow().isoformat(),
        },
    )
    # Set TTL so key doesn't linger forever
    redis_conn.expire(_job_key(job_id), JOB_TTL_SECONDS)

    # ✅ IMPORTANT: STRING PATH ONLY
    training_queue.enqueue(
        "workers.train_worker.run_training_job",
        kwargs={
            "job_id": job_id,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "problem_type": problem_type,
            "prep_config": prep_config,
            "user_params": user_params,
        },
        job_id=job_id,
    )

    return job_id


# =====================================================
# UPDATE JOB PROGRESS (CALLED BY WORKER)
# =====================================================

def update_job_progress(
    job_id: str,
    *,
    progress: int,
    message: Optional[str] = None,
):
    redis_conn.hset(
        _job_key(job_id),
        mapping={
            "status": "running",
            "progress": max(0, min(progress, 100)),
            "message": message,
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    redis_conn.expire(_job_key(job_id), JOB_TTL_SECONDS)


# =====================================================
# MARK JOB COMPLETED
# =====================================================

def mark_job_completed(
    job_id: str,
    *,
    model_id: int,
    metrics: Dict,
    model_file: str,
):
    redis_conn.hset(
        _job_key(job_id),
        mapping={
            "status": "completed",
            "progress": 100,
            "message": "Training completed",
            "result": json.dumps({
                "model_id": model_id,
                "metrics": metrics,
                "model_file": model_file,
            }),
            "completed_at": datetime.utcnow().isoformat(),
        },
    )
    redis_conn.expire(_job_key(job_id), JOB_TTL_SECONDS)


# =====================================================
# MARK JOB FAILED
# =====================================================

def mark_job_failed(job_id: str, *, error: str):
    redis_conn.hset(
        _job_key(job_id),
        mapping={
            "status": "failed",
            "message": error,
            "failed_at": datetime.utcnow().isoformat(),
        },
    )
    redis_conn.expire(_job_key(job_id), JOB_TTL_SECONDS)


# =====================================================
# GET JOB STATUS (UI POLLING)
# =====================================================

def get_job_status(job_id: str) -> Dict:
    # ── 1. Check our custom Redis hash first ──────────────────────────────
    if redis_conn.exists(_job_key(job_id)):
        data = redis_conn.hgetall(_job_key(job_id))
        result = data.get("result")
        return {
            "job_id": job_id,
            "status": data.get("status"),
            "progress": int(data.get("progress", 0)),
            "message": data.get("message"),
            "algorithm": data.get("algorithm"),
            "result": json.loads(result) if result else None,
        }

    # ── 2. Fallback: check RQ's own job registry ──────────────────────────
    # This handles the race condition where the worker finishes before
    # our custom key is written, or the key has expired.
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        rq_status = str(job.get_status())

        status_map = {
            "queued":   "queued",
            "started":  "running",
            "finished": "completed",
            "failed":   "failed",
            "stopped":  "failed",
            "canceled": "failed",
            "deferred": "queued",
        }

        mapped = status_map.get(rq_status, "running")

        return {
            "job_id":    job_id,
            "status":    mapped,
            "progress":  100 if mapped == "completed" else 50,
            "message":   f"Job {rq_status}",
            "algorithm": None,
            "result":    None,
        }
    except Exception:
        pass

    # ── 3. Truly not found ────────────────────────────────────────────────
    return {
        "status":   "not_found",
        "progress": 0,
        "message":  "Job not found",
    }


# =====================================================
# LIST JOBS (ADMIN / DEBUG)
# =====================================================

def list_jobs() -> List[Dict]:
    jobs = []

    for key in redis_conn.scan_iter("job:*"):
        data = redis_conn.hgetall(key)
        jobs.append({
            "job_id":     key.replace("job:", ""),
            "status":     data.get("status"),
            "progress":   int(data.get("progress", 0)),
            "message":    data.get("message"),
            "algorithm":  data.get("algorithm"),
            "created_at": data.get("created_at"),
        })

    return jobs
