"""Prediction Error Correction mechanism.

Computes the discrepancy between the current field activations and the
cognition state's predictions, then proposes a corrective field delta
weighted by epistemic confidence.
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import Contribution, CognitionDelta, FieldDelta
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class PredictionErrorCorrectionMechanism(BaseMechanism):
    """Minimises prediction error via gradient-like field correction.

    Parameters
    ----------
    eta:
        Learning rate (step size for error correction).
    update_predictions:
        Whether to also update cognition predictions toward actuals.
    """

    def __init__(self, eta: float = 0.1, update_predictions: bool = True) -> None:
        super().__init__(
            mechanism_id="prediction_error_correction",
            version="0.1.0",
            description="Corrects field activations toward predicted values.",
        )
        if not (0.0 < eta <= 1.0):
            raise ValueError(f"eta must be in (0, 1]; got {eta}")
        self.eta = eta
        self.update_predictions = update_predictions

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        field_vals = snapshot.field_state.values          # (n, f)
        predictions = snapshot.cognition.predictions       # (n, f)
        epistemic = snapshot.uncertainty.epistemic         # (n,)

        # Prediction error: how far the field deviates from expectations
        error = field_vals - predictions                   # (n, f)

        # Confidence: 1 - epistemic uncertainty
        confidence_weights = (1.0 - epistemic)[:, None]   # (n, 1)

        # Corrective delta: pull field toward prediction where we're confident
        field_delta = -self.eta * error * confidence_weights

        # Also update predictions toward actual values (online update)
        cog_delta = None
        if self.update_predictions:
            pred_update = self.eta * error                  # chase the actual
            attn_delta = np.zeros_like(snapshot.cognition.attention)
            cog_delta = CognitionDelta(
                delta_attention=attn_delta,
                delta_predictions=pred_update,
                weight=0.5,
            )

        error_norm = float(np.linalg.norm(np.nan_to_num(error, nan=0.0)))
        activation = float(np.clip(
            error_norm / (float(np.abs(np.nan_to_num(field_vals)).max()) + 1e-9), 0.0, 1.0
        ))

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=float(confidence_weights.mean()),
            activation=activation,
            estimated_cost=float(np.abs(field_delta).sum()),
            rationale=(
                f"Prediction error correction; |error|={error_norm:.4g}; "
                f"eta={self.eta}"
            ),
            field_delta=FieldDelta(delta=field_delta, weight=1.0),
            cognition_delta=cog_delta,
            diagnostics={
                "error_norm": error_norm,
                "eta": self.eta,
                "mean_confidence": float(confidence_weights.mean()),
            },
        )
