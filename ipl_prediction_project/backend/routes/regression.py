# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: regression.py
# Path: backend/routes/regression.py
# Description: FastAPI router exposing regression endpoints (training, evaluation, and real-time inference).
# ==============================================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.regression_service import train_and_evaluate, make_prediction

router = APIRouter(prefix="/regression", tags=["Regression"])

VALID_PROBLEMS   = {"first_inning_score", "batsman_score"}
VALID_ALGORITHMS = {"linear", "ridge", "lasso", "decision_tree"}

class TrainRequest(BaseModel):
    problem_type: str
    algorithms:   List[str]
    session_id:   str = None

class PredictRequest(BaseModel):
    problem_type: str
    algorithm:    str
    inputs:       Dict[str, Any]

@router.post("/train")
def train_regression(req: TrainRequest):
    # [Function]: train_regression
    # [Description]: Implements and executes the Train Regression logic within this module pipeline.
    if req.problem_type not in VALID_PROBLEMS:
        raise HTTPException(400, f"problem_type must be one of {VALID_PROBLEMS}")
    bad = [a for a in req.algorithms if a not in VALID_ALGORITHMS]
    if bad:
        raise HTTPException(400, f"Unknown algorithm(s): {bad}. Valid: {VALID_ALGORITHMS}")
    if not req.algorithms:
        raise HTTPException(400, "At least one algorithm must be selected.")
    try:
        result = train_and_evaluate(req.problem_type, req.algorithms, req.session_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Training error: {str(e)}")

@router.post("/predict")
def predict_regression(req: PredictRequest):
    # [Function]: predict_regression
    # [Description]: Implements and executes the Predict Regression logic within this module pipeline.
    if req.problem_type not in VALID_PROBLEMS:
        raise HTTPException(400, f"problem_type must be one of {VALID_PROBLEMS}")
    if req.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(400, f"algorithm must be one of {VALID_ALGORITHMS}")
    try:
        pred = make_prediction(req.problem_type, req.algorithm, req.inputs)
        return {"status": "success", "prediction": pred}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")
