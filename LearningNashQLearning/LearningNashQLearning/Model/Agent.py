import numpy as np
from .Environment import Environment, Game
from typing import Type, NewType


# Every player has an associaterd Q-table, implemented using the QTable class
# It is used by the agent to make decisions and learn by updating the values inside it, based on the actions taken and the rewards received
class QTable:
    def __init__(self, environment: Environment) -> None:
        pass


# The agent is the player that plays the game
# It can either get an action from the Q-Table, based on the values inside it in the current state, or learn by updating the Q-Table based on the rewards received
class Agent:
    def __init__(self):
        pass

    def getAction(self, game: Type[Game]) -> np.ndarray:
        pass

    def learn(self, startingGame: Type[Game], endingGame: Type[Game], action: np.ndarray, reward: np.ndarray) -> None:
        pass
