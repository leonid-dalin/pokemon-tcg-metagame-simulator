# src/api/models.py
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Union, List


class ExactSpec(BaseModel):
    exact: float = Field(..., ge=0.0, le=1.0)


class RangeSpec(BaseModel):
    min: float = Field(..., ge=0.0, le=1.0)
    max: float = Field(..., ge=0.0, le=1.0)


class PredictionRequest(BaseModel):
    job_id: str = Field(default="unknown")
    tournament_style: str = Field(default="championship_series")

    # Maps to your UserMetaSpec
    user_meta_spec: Dict[str, Union[float, ExactSpec, RangeSpec]] = Field(default_factory=dict)
    total_players: int = Field(default=256, ge=2, le=8192)
    min_sample_threshold: int = Field(default=10, ge=1)
    match_format: str = Field(default="BO3", pattern="^(BO1|BO3)$")

    # Monte Carlo specifics
    mc_iterations: int = Field(default=10_000)
    use_tie_convergence: bool = Field(default=True)
    global_tie_rate: float = Field(default=0.15)
    use_drop_feature: bool = Field(default=False)


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