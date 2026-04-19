# src/api/models.py
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Dict, Union, List, TypedDict, Any
from enum import Enum

class PrecisionTier(str, Enum):
    BULLET = "1 - BULLET"
    BLITZ = "2 - BLITZ"
    STANDARD = "3 - STANDARD"
    EXHAUSTIVE = "4 - EXHAUSTIVE"
    MAXIMUM = "5 - MAXIMUM"

TIER_MAPPING = {
    PrecisionTier.BULLET: 1_000,
    PrecisionTier.BLITZ: 10_000,
    PrecisionTier.STANDARD: 25_000,
    PrecisionTier.EXHAUSTIVE: 100_000,
    PrecisionTier.MAXIMUM: 250_000,
}

GLOBAL_TIE_RATE : float = 0.15

class ExactSpec(BaseModel):
    exact: float = Field(ge=0.0, le=1.0, description="Exact field presence between 0 and 1")

class RangeSpec(BaseModel):
    min: float = Field(ge=0.0, le=1.0)
    max: float = Field(ge=0.0, le=1.0)

class DeckRecommendation(TypedDict):
    deck: str
    expected_win_rate: float
    confidence: float
    sample_support: float
    meta_share: float
    is_user_specified: bool
    power_score: float
    frequency_score: float
    base_meta_score: float

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid") # Prevents mass-assignment injection attacks

    # 1. Identity & Metadata
    job_id: str = Field(default="unknown")

    # 2. Required Core Data
    deck_names: List[str]
    matchup_matrix: List[List[float]]

    # 3. Primary Tournament Parameters
    tournament_style: str = Field(default="pure_swiss")
    match_format: str = Field(default="BO3", pattern="^(BO1|BO3)$")
    total_players: int = Field(
        default=256,
        ge=8,
        le=8192,
        description="Total players in the field."
    )

    # 4. Meta & Field Constraints
    user_meta_spec: Dict[str, Union[float, ExactSpec, RangeSpec]] = Field(
        default_factory=dict,
        max_length=100,
    )
    meta_constraints: Dict[str, Union[ExactSpec, RangeSpec]] = Field(
        default_factory=dict
    )

    # 5. Simulation & Numerical Settings
    precision_tier: PrecisionTier = Field(
        default=PrecisionTier.STANDARD,
        ge=PrecisionTier.BULLET,
        le=PrecisionTier.MAXIMUM,
        description="Number of Monte Carlo brackets to simulate."
    )
    global_tie_rate: float = Field(
        default=GLOBAL_TIE_RATE,
        ge=0.0,
        le=0.5,
    )
    min_sample_threshold: int = Field(default=10, ge=1, le=100)

    # 6. Feature Flags (Booleans)
    use_tie_convergence: bool = Field(default=True)
    use_drop_feature: bool = Field(default=False)

class PredictionResult(TypedDict):
    recommendations: List[DeckRecommendation]
    avoid: List[DeckRecommendation]
    full_meta: Dict[str, float]
    metrics_per_deck: Dict[str, Any]
    swiss_rounds: int
    total_players: int
    frontrunners: List[str]

class MatchupStats(BaseModel):
    win_rate: float = Field(..., ge=0.0, le=1.0)
    match_count: int = Field(..., ge=0)

class ArchetypeMatchups(BaseModel):
    archetype_name: str = Field(..., min_length=1)
    # Dictionary mapping opponent archetype names to their matchup stats
    matchups: Dict[str, MatchupStats]

class ScrapedMatrix(BaseModel):
    format_name: str
    archetypes: List[ArchetypeMatchups]

    @model_validator(mode='after')
    def enforce_thermodynamic_purity(self):
        """
        Validates the entire matrix to ensure data purity rules are met before
        saving it to ea_input.json.
        """
        archetype_names = {a.archetype_name for a in self.archetypes}

        for archetype in self.archetypes:
            for opponent, stats in archetype.matchups.items():
                win_rate = stats.win_rate

                # 1. Ensure win rates are valid percentages
                if not (0.0 <= win_rate <= 1.0):
                    raise ValueError(f"Invalid win rate {win_rate} for {archetype.archetype_name} vs {opponent}")

                # 2. Ensure the diagonal (mirror match) is strictly 0.5
                if archetype.archetype_name == opponent and win_rate != 0.5:
                    raise ValueError(
                        f"Thermodynamic violation: {archetype.archetype_name} mirror match must be 0.5, got {win_rate}")

                # 3. Ensure no rogue strings or missing opponents
                if opponent not in archetype_names:
                    raise ValueError(f"Unrecognized opponent archetype: {opponent}")

        return self