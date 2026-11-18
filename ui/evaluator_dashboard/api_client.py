"""HTTP API client for Laptop 2 (Evaluator UI) communicating with Laptop 1 (FastAPI backend).

Expected Laptop 1 FastAPI endpoints (configurable via APIEndpoints):
  - GET  /health          -> {"status": "ok", "model_name": "...", "decoder": "hmm_viterbi"}
  - GET  /trips           -> {"trips": [{"trip_id": "...", "num_observations": ...}, ...]}
  - POST /predict         -> Accepts {"trip_id": "..."} and returns predicted route,
                             GPS trajectory, server-computed evaluation metrics, and continuity.

Ground-truth labels remain strictly server-side on Laptop 1 and are never requested or displayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


@dataclass(frozen=True)
class APIEndpoints:
    """Centralized endpoint paths so UI aligns instantly with Laptop 1 FastAPI routes."""

    health: str = "/health"
    trips: str = "/trips"
    predict: str = "/predict"


class APIClientError(RuntimeError):
    """Structured error raised when Laptop 1 API is unreachable or returns an error."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PathSnapAPIClient:
    """Client wrapper for Laptop 1 FastAPI inference and evaluation endpoints."""

    def __init__(
        self,
        base_url: str,
        endpoints: APIEndpoints = APIEndpoints(),
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoints = endpoints
        self.timeout_s = timeout_s

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def check_health(self) -> Dict[str, Any]:
        """Query GET /health to verify Laptop 1 connectivity."""
        try:
            resp = requests.get(self._url(self.endpoints.health), timeout=5.0)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as exc:
            raise APIClientError(
                f"Cannot connect to Laptop 1 API at {self.base_url}. Ensure FastAPI is running and reachable on the LAN."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise APIClientError(f"Health check timed out for {self.base_url}.") from exc
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            raise APIClientError(f"API health check failed (HTTP {code}): {exc}", status_code=code) from exc
        except ValueError as exc:
            raise APIClientError("API health endpoint returned invalid JSON.") from exc

    def list_trips(self) -> List[Dict[str, Any]]:
        """Dynamically fetch the available evaluation trip options from Laptop 1."""
        try:
            resp = requests.get(self._url(self.endpoints.trips), timeout=15.0)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.RequestException as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            raise APIClientError(f"Failed to fetch trip list from API: {exc}", status_code=code) from exc
        except ValueError as exc:
            raise APIClientError("GET /trips returned invalid JSON.") from exc

        # Support either {"trips": [...]} or a top-level list [...]
        raw_trips = payload.get("trips", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_trips, list):
            raise APIClientError("Unexpected schema from GET /trips: expected a list of trips.")

        normalized: List[Dict[str, Any]] = []
        for item in raw_trips:
            if isinstance(item, str):
                normalized.append({"trip_id": item})
            elif isinstance(item, dict) and "trip_id" in item:
                normalized.append(item)
        return normalized

    def run_prediction(self, trip_id: str) -> Dict[str, Any]:
        """Trigger server-side prediction & evaluation on Laptop 1 for `trip_id`."""
        try:
            resp = requests.post(
                self._url(self.endpoints.predict),
                json={"trip_id": trip_id},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout as exc:
            raise APIClientError(
                f"Prediction request timed out after {self.timeout_s:.0f}s for trip '{trip_id}'."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            detail = ""
            try:
                detail = f" — {exc.response.json().get('detail', '')}"
            except Exception:
                pass
            raise APIClientError(
                f"Prediction failed for trip '{trip_id}' (HTTP {code}){detail}", status_code=code
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise APIClientError(f"Network error calling POST /predict: {exc}") from exc
        except ValueError as exc:
            raise APIClientError("POST /predict returned invalid JSON.") from exc

        # Defensive guard: never allow raw ground-truth route sequences into the UI state
        data.pop("reference_route", None)
        data.pop("ground_truth", None)
        data.pop("ground_truth_arcs", None)
        return data
