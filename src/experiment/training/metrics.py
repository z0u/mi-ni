from pydantic import BaseModel, NonNegativeInt


class TrainingMetrics(BaseModel, validate_assignment=True):
    epoch: NonNegativeInt
    learning_rate: float
    val_loss: float
    training_tokens: NonNegativeInt
