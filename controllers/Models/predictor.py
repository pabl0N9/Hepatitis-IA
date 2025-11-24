from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_regresion_logistica.pkl"
SCALER_PATH = BASE_DIR / "scaler (2).pkl"

FALLBACK_FEATURE_ORDER = [
    "Age",
    "Sex",
    "Steroid",
    "Antivirals",
    "Fatigue",
    "Malaise",
    "Anorexia",
    "Liver_Big",
    "Liver_Firm",
    "Spleen_Palpable",
    "Spiders",
    "Ascites",
    "Varices",
    "Bilirubin",
    "Alk_Phosphate",
    "Sgot",
    "Albumin",
    "Protime",
    "Histology",
    "Ciudad",
    "Estado_Civil",
]

BINARY_FEATURES = {
    "Steroid",
    "Antivirals",
    "Fatigue",
    "Malaise",
    "Anorexia",
    "Liver_Big",
    "Liver_Firm",
    "Spleen_Palpable",
    "Spiders",
    "Ascites",
    "Varices",
    "Histology",
}
BINARY_YES_VALUE = 1
BINARY_NO_VALUE = 2

PREDICTION_LABELS = {
    "1": "Muere",
    "2": "Vive",
}


class ArtifactLoadError(Exception):
    """Raised when model artifacts are not available."""


class HepatitisPredictor:
    """Loads hepatitis model artifacts and serves predictions."""

    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: Any = None
        self.feature_order: List[str] = list(FALLBACK_FEATURE_ORDER)
        self.startup_error: Optional[Exception] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            try:
                self.feature_order = list(self.scaler.feature_names_in_)  # type: ignore[attr-defined]
            except Exception:
                self.feature_order = list(FALLBACK_FEATURE_ORDER)
        except Exception as exc:  # pragma: no cover - startup safety
            self.startup_error = exc

    @property
    def ready(self) -> bool:
        return self.startup_error is None and self.model is not None and self.scaler is not None

    def schema(self) -> Dict[str, Any]:
        return {
            "expected_features": self.feature_order,
            "example_payload": self.example_payload(),
            "binary_features": sorted(BINARY_FEATURES),
        }

    def example_payload(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        for feature in self.feature_order:
            if feature in BINARY_FEATURES:
                payload[feature] = float(BINARY_NO_VALUE)
            else:
                payload[feature] = 0.0
        return payload

    def _to_scaled_array(self, payload: Dict[str, Any]) -> np.ndarray:
        if not self.ready:
            raise ArtifactLoadError("Artifacts are not loaded")

        values: List[float] = []
        for feature in self.feature_order:
            if feature not in payload:
                raise KeyError(feature)
            try:
                values.append(float(payload[feature]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {feature} must be numeric") from exc

        raw = np.array([values], dtype=float)
        return self.scaler.transform(raw)

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scaled = self._to_scaled_array(payload)
        prediction = self.model.predict(scaled)[0]

        probabilities: Dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            try:
                scores = self.model.predict_proba(scaled)[0]
                labels = getattr(self.model, "classes_", [])
                probabilities = {str(label): float(score) for label, score in zip(labels, scores)}
            except Exception:
                probabilities = {}

        return {
            "prediction": str(prediction),
            "prediction_label": PREDICTION_LABELS.get(str(prediction), "Desconocido"),
            "probabilities": probabilities,
            "feature_order": self.feature_order,
        }
