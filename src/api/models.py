# src/api/models.py
from pydantic import BaseModel, Field
from typing import Dict, Union, Optional

class ExactSpec(BaseModel):
    exact: float = Field(..., ge=0.0, le=1.0)

class RangeSpec(BaseModel):
    min: float = Field(..., ge=0.0, le=1.0)
    max: float = Field(..., ge=0.0, le=1.0)


class PredictionRequest(BaseModel):
    # Maps to your UserMetaSpec
    user_meta_spec: Dict[str, Union[float, ExactSpec, RangeSpec]] = Field(default_factory=dict)
    total_players: int = Field(default=256, ge=2, le=8192)
    min_sample_threshold: int = Field(default=10, ge=1)
    match_format: str = Field(default="BO3", pattern="^(BO1|BO3)$")

    # Monte Carlo specifics
    mc_iterations: int = Field(default=10000)
    use_tie_convergence: bool = Field(default=True)
    global_tie_rate: float = Field(default=0.15)
    use_drop_feature: bool = Field(default=False)