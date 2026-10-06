# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: unsupervised.py
# Path: backend/routes/unsupervised.py
# Description: FastAPI router exposing unsupervised player clustering (K-Means/DBSCAN) and similar player search.
# ==============================================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.unsupervised_service import (
    train_and_evaluate_clustering, 
    find_similar_players,
    get_player_stats
)

router = APIRouter(prefix="/unsupervised", tags=["Unsupervised"])

VALID_PROBLEMS = {"batsman", "bowler"}
VALID_ALGORITHMS = {"kmeans", "dbscan"}

class TrainRequest(BaseModel):
    problem_type: str
    algorithm: str
    params: Dict[str, Any] = {}

class SimilarRequest(BaseModel):
    problem_type: str
    algorithm: str
    player_name: str
    top_n: int = 5

@router.post("/train")
def train_clustering(req: TrainRequest):
    # [Function]: train_clustering
    # [Description]: Implements and executes the Train Clustering logic within this module pipeline.
    if req.problem_type not in VALID_PROBLEMS:
        raise HTTPException(400, f"problem_type must be one of {VALID_PROBLEMS}")
    if req.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(400, f"algorithm must be one of {VALID_ALGORITHMS}")
    try:
        res = train_and_evaluate_clustering(req.problem_type, req.algorithm, req.params)
        return res
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Clustering error: {str(e)}")

@router.get("/players")
def get_players(problem_type: str):
    # [Function]: get_players
    # [Description]: Implements and executes the Get Players logic within this module pipeline.
    if problem_type not in VALID_PROBLEMS:
        raise HTTPException(400, f"problem_type must be one of {VALID_PROBLEMS}")
    try:
        df = get_player_stats(problem_type)
        players = sorted(df['player_name'].tolist())
        return {"players": players}
    except Exception as e:
        raise HTTPException(500, f"Error getting players: {str(e)}")

@router.post("/similar")
def get_similar(req: SimilarRequest):
    # [Function]: get_similar
    # [Description]: Implements and executes the Get Similar logic within this module pipeline.
    if req.problem_type not in VALID_PROBLEMS:
        raise HTTPException(400, f"problem_type must be one of {VALID_PROBLEMS}")
    if req.algorithm not in VALID_ALGORITHMS:
        raise HTTPException(400, f"algorithm must be one of {VALID_ALGORITHMS}")
    try:
        res = find_similar_players(req.problem_type, req.algorithm, req.player_name, req.top_n)
        return res
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error finding similar players: {str(e)}")
