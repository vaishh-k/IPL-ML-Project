# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: ensemble.py
# Path: backend/routes/ensemble.py
# Description: FastAPI router exposing ensemble learning pipelines (RF, AdaBoost, XGBoost, GB) for regression/classification.
# ==============================================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.ensemble_service import (
    train_and_evaluate_ensemble_reg,
    make_ensemble_reg_prediction,
    train_and_evaluate_ensemble_cls,
    make_ensemble_cls_prediction
)

router = APIRouter(prefix="/ensemble", tags=["Ensemble Learning"])

VALID_ALGORITHMS = {"rf", "gb", "xgb", "ada"}

class TrainRequest(BaseModel):
    algorithms: List[str]
    session_id: str = None

class PredictRequest(BaseModel):
    algorithm: str
    inputs: Dict[str, Any]

@router.post("/train/regression")
def train_reg(req: TrainRequest):
    # [Function]: train_reg
    # [Description]: Implements and executes the Train Reg logic within this module pipeline.
    bad = [a for a in req.algorithms if a not in VALID_ALGORITHMS]
    if bad:
        raise HTTPException(400, f"Unknown algorithm(s): {bad}. Valid: {VALID_ALGORITHMS}")
    if not req.algorithms:
        raise HTTPException(400, "At least one algorithm must be selected.")
    try:
        res = train_and_evaluate_ensemble_reg(req.algorithms, req.session_id)
        return res
    except Exception as e:
        raise HTTPException(500, f"Ensemble regression training error: {str(e)}")

@router.post("/predict/regression")
def predict_reg(req: PredictRequest):
    # [Function]: predict_reg
    # [Description]: Implements and executes the Predict Reg logic within this module pipeline.
    if req.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(400, f"Invalid algorithm: {req.algorithm}. Valid: {VALID_ALGORITHMS}")
    try:
        res = make_ensemble_reg_prediction(req.algorithm, req.inputs)
        return {"status": "success", **res}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ensemble regression prediction error: {str(e)}")

@router.post("/train/classification")
def train_cls(req: TrainRequest):
    # [Function]: train_cls
    # [Description]: Implements and executes the Train Cls logic within this module pipeline.
    bad = [a for a in req.algorithms if a not in VALID_ALGORITHMS]
    if bad:
        raise HTTPException(400, f"Unknown algorithm(s): {bad}. Valid: {VALID_ALGORITHMS}")
    if not req.algorithms:
        raise HTTPException(400, "At least one algorithm must be selected.")
    try:
        res = train_and_evaluate_ensemble_cls(req.algorithms, req.session_id)
        return res
    except Exception as e:
        raise HTTPException(500, f"Ensemble classification training error: {str(e)}")

@router.post("/predict/classification")
def predict_cls(req: PredictRequest):
    # [Function]: predict_cls
    # [Description]: Implements and executes the Predict Cls logic within this module pipeline.
    if req.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(400, f"Invalid algorithm: {req.algorithm}. Valid: {VALID_ALGORITHMS}")
    try:
        res = make_ensemble_cls_prediction(req.algorithm, req.inputs)
        return {"status": "success", **res}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ensemble classification prediction error: {str(e)}")
