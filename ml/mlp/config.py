"""Configuration for the PathSnap MLP baseline pipeline (Person 4)."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple
import json


# Workspace paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE_ROOT / "Student folder" / "Student folder"
TRAIN_DIR = DATA_ROOT / "train"
STUDENT_TEST_DIR = DATA_ROOT / "student_test"

MLP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MLP_DIR / "artifacts"
RESULTS_DIR = MLP_DIR / "results"
PLOTS_DIR = MLP_DIR / "plots"
CACHE_DIR = MLP_DIR / "cache"

API_INTEGRATION_DIR = WORKSPACE_ROOT / "api_integration"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
UI_DIR = WORKSPACE_ROOT / "ui" / "evaluator_dashboard"


DEFAULT_FEATURE_COLUMNS: List[str] = [
    # Spatial & projection geometry
    "perp_dist_m",
    "log_perp_dist",
    "proj_ratio",
    "is_interior_proj",
    "dist_to_start_m",
    "dist_to_end_m",
    "arc_length_m",
    "log_arc_length",
    # Directional / heading alignment
    "cos_heading_diff",
    "sin_heading_diff_abs",
    "heading_diff_norm",
    "heading_valid",
    # Local vehicle kinematics
    "speed_mps",
    "dt_prev_clipped",
    "is_time_gap",
    # Local temporal context
    "prev_obs_dist_m",
    "next_obs_dist_m",
    "progress_along_arc",
    "is_prev_connected",
    "is_next_connected",
    # Road-network topology
    "has_reverse_arc",
    "from_node_deg",
    "to_node_deg",
    # Within-observation competitive context
    "dist_rank_norm",
    "dist_diff_from_min_m",
    "dist_ratio_to_min",
    "cos_heading_diff_vs_best",
    "num_candidates_obs",
]


@dataclass
class CandidateConfig:
    """Candidate generation and alignment configuration."""

    max_candidates: int = 12
    search_radius_m: float = 75.0
    fallback_radius_m: float = 150.0
    segment_sample_step_m: float = 25.0
    align_heading_weight_m: float = 12.0
    align_step_penalty_m: float = 0.5
    time_gap_threshold_s: float = 5.0
    max_speed_mps: float = 50.0


@dataclass
class MLPConfig:
    """Configurable MLP architecture and training hyperparameters."""

    experiment_name: str = "mlp_128_64_32"
    hidden_dims: Tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.20
    use_batch_norm: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    epochs: int = 25
    early_stopping_patience: int = 5
    pos_weight_mode: str = "none"  # "none" (calibrated probabilities) or "balanced"
    n_cv_folds: int = 5
    random_seed: int = 42
    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MLPConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["hidden_dims"] = tuple(data["hidden_dims"])
        return cls(**data)


def ensure_directories() -> None:
    """Create standard output directories if absent."""
    for d in (ARTIFACTS_DIR, RESULTS_DIR, PLOTS_DIR, CACHE_DIR, API_INTEGRATION_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
