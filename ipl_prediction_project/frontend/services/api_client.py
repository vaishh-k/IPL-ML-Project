# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: api_client.py
# Path: frontend/services/api_client.py
# Description: HTTP API client connecting the Streamlit frontend with the FastAPI backend services.
# ==============================================================================

import os
import httpx
import streamlit as st

BASE_URL = os.environ.get("IPL_API_URL", "http://127.0.0.1:8001")

class ApiClient:
    def __init__(self, session_id: str):
        # [Function]: __init__
        # [Description]: Implements and executes the   Init   logic within this module pipeline.
        self.session_id = session_id
        self.client = httpx.Client(base_url=BASE_URL, timeout=300.0)

    # Cache helpers
    def _cache_key(self, endpoint: str, params: dict) -> str:
        # [Function]: _cache_key
        # [Description]: Implements and executes the  Cache Key logic within this module pipeline.
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "session_id")
        return f"{self.session_id}::{endpoint}?{param_str}"

    def _read_cache(self, key: str):
        # [Function]: _read_cache
        # [Description]: Implements and executes the  Read Cache logic within this module pipeline.
        return st.session_state.get("api_cache", {}).get(key)

    def _write_cache(self, key: str, value: dict):
        # [Function]: _write_cache
        # [Description]: Implements and executes the  Write Cache logic within this module pipeline.
        if "api_cache" not in st.session_state:
            st.session_state["api_cache"] = {}
        st.session_state["api_cache"][key] = value

    def clear_session_cache(self):
        # [Function]: clear_session_cache
        # [Description]: Implements and executes the Clear Session Cache logic within this module pipeline.
        cache = st.session_state.get("api_cache", {})
        stale = [k for k in list(cache.keys()) if k.startswith(self.session_id + "::")]
        for k in stale:
            del cache[k]

    # HTTP helpers
    def _parse_error(self, response, exception) -> str:
        # [Function]: _parse_error
        # [Description]: Implements and executes the  Parse Error logic within this module pipeline.
        try:
            return response.json().get("detail", str(exception))
        except Exception:
            return response.text or str(exception)

    def _get(self, endpoint: str, params: dict = None, use_cache: bool = True) -> dict:
        # [Function]: _get
        # [Description]: Implements and executes the  Get logic within this module pipeline.
        if params is None:
            params = {}
        cache_key = self._cache_key(endpoint, params)
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached
        req_params = {**params, "session_id": self.session_id}
        try:
            response = self.client.get(endpoint, params=req_params)
            response.raise_for_status()
            result = response.json()
            if use_cache and not result.get("error"):
                self._write_cache(cache_key, result)
            return result
        except httpx.HTTPStatusError as e:
            err_detail = self._parse_error(response, e)
            return {"error": True, "detail": err_detail, "status_code": response.status_code}
        except Exception as e:
            return {"error": True, "detail": str(e)}

    def _post(self, endpoint: str, params: dict = None, json_body: dict = None, files: dict = None) -> dict:
        # [Function]: _post
        # [Description]: Implements and executes the  Post logic within this module pipeline.
        if params is None:
            params = {}
        req_params = {**params, "session_id": self.session_id}
        try:
            if files:
                response = self.client.post(endpoint, params=req_params, files=files)
            else:
                response = self.client.post(endpoint, params=req_params, json=json_body)
            response.raise_for_status()
            result = response.json()
            self.clear_session_cache()
            return result
        except httpx.HTTPStatusError as e:
            err_detail = self._parse_error(response, e)
            st.error(f"API Error ({response.status_code}): {err_detail}")
            return {"error": True, "detail": err_detail}
        except Exception as e:
            st.error(f"Failed to connect to backend: {str(e)}")
            return {"error": True, "detail": str(e)}

    # Dataset Endpoints
    def list_local_datasets(self) -> dict:
        # [Function]: list_local_datasets
        # [Description]: Implements and executes the List Local Datasets logic within this module pipeline.
        return self._get("/dataset/list-local", use_cache=False)

    def select_local_dataset(self, filename: str) -> dict:
        # [Function]: select_local_dataset
        # [Description]: Implements and executes the Select Local Dataset logic within this module pipeline.
        return self._post("/dataset/select-local", params={"filename": filename})

    def save_processed_dataset(self, filename: str) -> dict:
        # [Function]: save_processed_dataset
        # [Description]: Implements and executes the Save Processed Dataset logic within this module pipeline.
        return self._post("/dataset/save-processed", params={"filename": filename})

    def get_preview(self, stage: str = None, rows_type: str = "first", limit: int = 10, use_cache: bool = True) -> dict:
        # [Function]: get_preview
        # [Description]: Implements and executes the Get Preview logic within this module pipeline.
        params = {"rows_type": rows_type, "limit": limit}
        if stage:
            params["stage"] = stage
        return self._get("/dataset/preview", params=params, use_cache=use_cache)

    # EDA Endpoints
    def get_overview(self) -> dict:
        # [Function]: get_overview
        # [Description]: Implements and executes the Get Overview logic within this module pipeline.
        return self._get("/eda/overview")

    def get_quality_check(self) -> dict:
        # [Function]: get_quality_check
        # [Description]: Implements and executes the Get Quality Check logic within this module pipeline.
        return self._get("/eda/quality-check")

    def get_missing_values(self) -> dict:
        # [Function]: get_missing_values
        # [Description]: Implements and executes the Get Missing Values logic within this module pipeline.
        return self._get("/eda/missing-values")

    def clean_missing_values(self, strategies: dict) -> dict:
        # [Function]: clean_missing_values
        # [Description]: Implements and executes the Clean Missing Values logic within this module pipeline.
        return self._post("/eda/clean-missing", json_body={"strategies": strategies})

    def get_duplicates(self) -> dict:
        # [Function]: get_duplicates
        # [Description]: Implements and executes the Get Duplicates logic within this module pipeline.
        return self._get("/eda/duplicates")

    def remove_duplicates(self) -> dict:
        # [Function]: remove_duplicates
        # [Description]: Implements and executes the Remove Duplicates logic within this module pipeline.
        return self._post("/eda/remove-duplicates")

    def get_dtypes(self) -> dict:
        # [Function]: get_dtypes
        # [Description]: Implements and executes the Get Dtypes logic within this module pipeline.
        return self._get("/eda/dtypes")

    def get_univariate(self, column: str) -> dict:
        # [Function]: get_univariate
        # [Description]: Implements and executes the Get Univariate logic within this module pipeline.
        return self._get("/eda/univariate", params={"column": column})

    def get_distribution(self) -> dict:
        # [Function]: get_distribution
        # [Description]: Implements and executes the Get Distribution logic within this module pipeline.
        return self._get("/eda/distribution")

    def get_outliers(self, column: str, method: str = "iqr") -> dict:
        # [Function]: get_outliers
        # [Description]: Implements and executes the Get Outliers logic within this module pipeline.
        return self._get("/eda/outliers", params={"column": column, "method": method})

    def handle_outliers(self, column: str, strategy: str, method: str = "iqr") -> dict:
        # [Function]: handle_outliers
        # [Description]: Implements and executes the Handle Outliers logic within this module pipeline.
        return self._post("/eda/handle-outliers", json_body={"column": column, "strategy": strategy, "method": method})

    def get_bivariate(
        self,
        feature_x: str,
        feature_y: str,
        show_trendline: bool = True,
        show_points: bool = True,
        agg_type: str = "mean",
        normalize_stack: bool = False,
        normalize_contingency: str = "none"
    ) -> dict:
        # [Function]: get_bivariate
        # [Description]: Implements and executes the Get Bivariate logic within this module pipeline.
        params = {
            "feature_x": feature_x,
            "feature_y": feature_y,
            "show_trendline": show_trendline,
            "show_points": show_points,
            "agg_type": agg_type,
            "normalize_stack": normalize_stack,
            "normalize_contingency": normalize_contingency
        }
        return self._get("/eda/bivariate", params=params)

    def get_correlation(self) -> dict:
        # [Function]: get_correlation
        # [Description]: Implements and executes the Get Correlation logic within this module pipeline.
        return self._get("/eda/correlation")

    def get_feature_selection(self, target: str = None) -> dict:
        # [Function]: get_feature_selection
        # [Description]: Implements and executes the Get Feature Selection logic within this module pipeline.
        params = {}
        if target:
            params["target"] = target
        return self._get("/eda/feature-selection", params=params, use_cache=False)

    # Preprocessing Endpoints
    def encode_categorical(self, encoding_map: dict) -> dict:
        # [Function]: encode_categorical
        # [Description]: Implements and executes the Encode Categorical logic within this module pipeline.
        return self._post("/preprocessing/encode", json_body={"encoding_map": encoding_map})

    def scale_numerical(self, scaling_map: dict) -> dict:
        # [Function]: scale_numerical
        # [Description]: Implements and executes the Scale Numerical logic within this module pipeline.
        return self._post("/preprocessing/scale", json_body={"scaling_map": scaling_map})

    def split_dataset(self, test_size: float = 0.2) -> dict:
        # [Function]: split_dataset
        # [Description]: Implements and executes the Split Dataset logic within this module pipeline.
        return self._post("/preprocessing/split", json_body={"test_size": test_size})

    # PCA Endpoints
    def run_pca(self, n_components: int = 2, target_col: str = None) -> dict:
        # [Function]: run_pca
        # [Description]: Implements and executes the Run Pca logic within this module pipeline.
        body = {"n_components": n_components}
        if target_col:
            body["target_col"] = target_col
        return self._post("/pca/run", json_body=body)

    def get_summary(self, target: str = None) -> dict:
        # [Function]: get_summary
        # [Description]: Implements and executes the Get Summary logic within this module pipeline.
        params = {}
        if target:
            params["target"] = target
        return self._get("/eda/summary", params=params)

    # Regression Endpoints
    def train_regression(self, problem_type: str, algorithms: list) -> dict:
        # [Function]: train_regression
        # [Description]: Implements and executes the Train Regression logic within this module pipeline.
        return self._post("/regression/train", json_body={"problem_type": problem_type, "algorithms": algorithms, "session_id": self.session_id})

    def predict_regression(self, problem_type: str, algorithm: str, inputs: dict) -> dict:
        # [Function]: predict_regression
        # [Description]: Implements and executes the Predict Regression logic within this module pipeline.
        return self._post("/regression/predict", json_body={"problem_type": problem_type, "algorithm": algorithm, "inputs": inputs})

    def train_classification(self, problem_type: str, algorithms: list) -> dict:
        # [Function]: train_classification
        # [Description]: Implements and executes the Train Classification logic within this module pipeline.
        return self._post("/classification/train", json_body={"problem_type": problem_type, "algorithms": algorithms, "session_id": self.session_id})

    def predict_classification(self, problem_type: str, algorithm: str, inputs: dict) -> dict:
        # [Function]: predict_classification
        # [Description]: Implements and executes the Predict Classification logic within this module pipeline.
        return self._post("/classification/predict", json_body={"problem_type": problem_type, "algorithm": algorithm, "inputs": inputs})

    # Unsupervised Endpoints
    def train_unsupervised(self, problem_type: str, algorithm: str, params: dict) -> dict:
        # [Function]: train_unsupervised
        # [Description]: Implements and executes the Train Unsupervised logic within this module pipeline.
        return self._post("/unsupervised/train", json_body={"problem_type": problem_type, "algorithm": algorithm, "params": params})

    def get_unsupervised_players(self, problem_type: str) -> dict:
        # [Function]: get_unsupervised_players
        # [Description]: Implements and executes the Get Unsupervised Players logic within this module pipeline.
        return self._get("/unsupervised/players", params={"problem_type": problem_type})

    def find_similar_players(self, problem_type: str, algorithm: str, player_name: str, top_n: int = 5) -> dict:
        # [Function]: find_similar_players
        # [Description]: Implements and executes the Find Similar Players logic within this module pipeline.
        return self._post("/unsupervised/similar", json_body={"problem_type": problem_type, "algorithm": algorithm, "player_name": player_name, "top_n": top_n})

    # Ensemble Endpoints
    def train_ensemble_reg(self, algorithms: list) -> dict:
        # [Function]: train_ensemble_reg
        # [Description]: Implements and executes the Train Ensemble Reg logic within this module pipeline.
        return self._post("/ensemble/train/regression", json_body={"algorithms": algorithms, "session_id": self.session_id})

    def predict_ensemble_reg(self, algorithm: str, inputs: dict) -> dict:
        # [Function]: predict_ensemble_reg
        # [Description]: Implements and executes the Predict Ensemble Reg logic within this module pipeline.
        return self._post("/ensemble/predict/regression", json_body={"algorithm": algorithm, "inputs": inputs})

    def train_ensemble_cls(self, algorithms: list) -> dict:
        # [Function]: train_ensemble_cls
        # [Description]: Implements and executes the Train Ensemble Cls logic within this module pipeline.
        return self._post("/ensemble/train/classification", json_body={"algorithms": algorithms, "session_id": self.session_id})

    def predict_ensemble_cls(self, algorithm: str, inputs: dict) -> dict:
        # [Function]: predict_ensemble_cls
        # [Description]: Implements and executes the Predict Ensemble Cls logic within this module pipeline.
        return self._post("/ensemble/predict/classification", json_body={"algorithm": algorithm, "inputs": inputs})
