"""HTTP API client for Laptop 2 (Streamlit Frontend) -> Laptop 1 (FastAPI Backend).

Uses configurable `PATHSNAP_API_URL` environment variable (no hardcoded laptop IPs).
Supports all Review 2 endpoints:
  - GET  /health
  - GET  /models
  - GET  /trips
  - GET  /trips/{trip_id}
  - POST /predict
  - POST /evaluate
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


@dataclass(frozen=True)
class APIEndpoints:
    """Centralized FastAPI endpoint paths."""

    health: str = "/health"
    models: str = "/models"
    trips: str = "/trips"
    predict: str = "/predict"
    evaluate: str = "/evaluate"


class APIClientError(RuntimeError):
    """Structured error raised when Laptop 1 API is unreachable or returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def get_default_api_url() -> str:
    """Read default FastAPI base URL from PATHSNAP_API_URL environment variable."""
    return os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")


class PathSnapAPIClient:
    """HTTP client wrapper for Laptop 1 FastAPI inference and evaluation endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoints: APIEndpoints = APIEndpoints(),
        timeout_s: float = 90.0,
    ) -> None:
        self.base_url = (base_url or get_default_api_url()).rstrip("/")
        self.endpoints = endpoints
        self.timeout_s = timeout_s

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _handle_request(self, method: str, path: str, timeout: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        url = self._url(path)
        t_out = timeout if timeout is not None else self.timeout_s
        try:
            resp = requests.request(method, url, timeout=t_out, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise APIClientError(
                f"Cannot connect to Laptop 1 FastAPI at {self.base_url}. "
                "Verify the server is running (`uvicorn api.main:app --host 0.0.0.0 --port 8000`) "
                "and both laptops are on the same network."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise APIClientError(f"Request to {url} timed out after {t_out:.0f}s.") from exc
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            detail = ""
            try:
                detail = f" — {exc.response.json().get('detail', '')}"
            except Exception:
                pass
            raise APIClientError(
                f"API request failed ({method} {path}, HTTP {code}){detail}", status_code=code
            ) from exc
        except ValueError as exc:
            raise APIClientError(f"Endpoint {method} {path} returned invalid JSON.") from exc

    def check_health(self) -> Dict[str, Any]:
        return self._handle_request("GET", self.endpoints.health, timeout=5.0)

    def list_models(self) -> Dict[str, Any]:
        return self._handle_request("GET", self.endpoints.models, timeout=15.0)

    def list_trips(self, partition: str = "student_test") -> List[Dict[str, Any]]:
        payload = self._handle_request(
            "GET", self.endpoints.trips, params={"partition": partition}, timeout=20.0
        )
        raw_trips = payload.get("trips", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_trips, list):
            raise APIClientError("Unexpected schema from GET /trips: expected a list.")
        return [
            {"trip_id": item} if isinstance(item, str) else item
            for item in raw_trips
            if isinstance(item, (str, dict))
        ]

    def get_trip_detail(self, trip_id: str, partition: str = "student_test") -> Dict[str, Any]:
        return self._handle_request(
            "GET",
            f"{self.endpoints.trips}/{trip_id}",
            params={"partition": partition},
            timeout=20.0,
        )

    def run_prediction(
        self,
        trip_id: str,
        model_name: str = "mlp_128_64_32",
        partition: str = "student_test",
        include_reference: bool = False,
        include_candidate_segments: bool = True,
        include_observations: bool = False,
    ) -> Dict[str, Any]:
        body = {
            "trip_id": trip_id,
            "partition": partition,
            "model_name": model_name,
            "include_reference": include_reference and (partition != "evaluator_test"),
            "include_candidate_segments": include_candidate_segments,
            "include_observations": include_observations,
        }
        data = self._handle_request("POST", self.endpoints.predict, json=body)
        if partition == "evaluator_test":
            data.pop("reference_segments", None)
        return data

    def run_evaluation(
        self,
        model_name: Optional[str] = None,
        trip_id: Optional[str] = None,
        partition: str = "student_test",
    ) -> Dict[str, Any]:
        body = {
            "model_name": model_name,
            "trip_id": trip_id,
            "partition": partition,
        }
        return self._handle_request("POST", self.endpoints.evaluate, json=body)
