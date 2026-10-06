# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: classification.py
# Path: backend/routes/classification.py
# Description: FastAPI router exposing classification endpoints (training, evaluation, and predictions).
# ==============================================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.classification_service import train_and_evaluate_cls, make_classification_prediction

router = APIRouter(prefix="/classification", tags=["Classification"])

VALID_PROBLEMS   = {"defend_target", "batsman_fifty"}
VALID_ALGORITHMS = {"logistic", "decision_tree", "svm", "naive_bayes"}

class TrainRequest(BaseModel):
    problem_type: str
    algorithms:   List[str]
    session_id:   str = None

class PredictRequest(BaseModel):
    problem_type: str
    algorithm:    str
    inputs:       Dict[str, Any]

@router.post("/train")
def train_classification(req: TrainRequest):
    # [Function]: train_classification
    # [Description]: Implements and executes the Train Classification logic within this module pipeline.
    if req.problem_type not in VALID_PROBLEMS:
        raise HTTPException(400, f"problem_type must be one of {VALID_PROBLEMS}")
    bad = [a for a in req.algorithms if a not in VALID_ALGORITHMS]
    if bad:
        raise HTTPException(400, f"Unknown algorithm(s): {bad}. Valid: {VALID_ALGORITHMS}")
    if not req.algorithms:
        raise HTTPException(400, "At least one algorithm must be selected.")
    try:
        result = train_and_evaluate_cls(req.problem_type, req.algorithms, req.session_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Training error: {str(e)}")

@router.post("/predict")
def predict_classification(req: PredictRequest):
    # [Function]: predict_classification
    # [Description]: Implements and executes the Predict Classification logic within this module pipeline.
    if req.problem_type not in VALID_PROBLEMS:
        raise HTTPException(400, f"problem_type must be one of {VALID_PROBLEMS}")
    if req.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(400, f"algorithm must be one of {VALID_ALGORITHMS}")
    try:
        result = make_classification_prediction(req.problem_type, req.algorithm, req.inputs)
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")
