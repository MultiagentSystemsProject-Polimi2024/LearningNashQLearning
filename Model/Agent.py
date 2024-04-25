import numpy as np
from Environment import Environment, Game
from typing import Type, NewType


class QTable:
    def __init__(self, environment: Environment) -> None:
        pass


class Agent:
    def __init__(self):
        pass

    def getAction(self, game: Type[Game]) -> np.ndarray:
        pass

    def learn(self, startingGame: Type[Game], endingGame: Type[Game], action: np.ndarray, reward: np.ndarray) -> None:
        pass
