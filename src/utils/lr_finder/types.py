from typing import Literal, TypeAlias
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator
from pydantic.dataclasses import dataclass


@dataclass
class Range:
    low: float
    high: float

    @model_validator(mode='after')
    def validate_bounds(self):
        assert self.high >= self.low, 'high >= low'
        return self


SearchMethod: TypeAlias = Literal['steepest', 'lowest', 'balanced']


@dataclass
class LRFinderConfig:
    num_zooms: PositiveInt = 5
    method: SearchMethod = 'steepest'
    zoom_factor: PositiveFloat = 0.5
    steps_per_zoom: PositiveInt = 10
    start_lr: NonNegativeFloat = 1e-10
    end_lr: NonNegativeFloat = 1e1


@dataclass
class LRFinderSeries:
    lrs: list[NonNegativeFloat]
    losses: list[NonNegativeFloat]
    best_lr: NonNegativeFloat
    steepest_lr: NonNegativeFloat
    zoom: int


@dataclass
class Progress:
    step: NonNegativeInt | None = None
    total_steps: NonNegativeInt | None = None
    info: dict | None = None
