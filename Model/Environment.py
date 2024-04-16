import numpy as np
from typing import Type, NewType


class TransitionProfile:
    transitions: dict = {}

    def __init__(self, transitions: dict) -> None:
        self.transitions = transitions

    def setTransition(self, nextGame, probability: float) -> None:
        self.transitions[nextGame] = probability

    def getTransitions(self) -> tuple:
        return self.transitions.keys(), self.transitions.values()

    def sampleTransition(self):
        nextGames, probabilities = self.getTransitions()
        return np.random.choice(nextGames, p=probabilities)

    def __str__(self) -> str:
        return str(self.transitions)


class Game:
    NPlayers: int = 1
    possibleActions: np.ndarray = np.array([1])
    transitionMatrix: np.ndarray
    payoffMatrix: np.ndarray = np.array([.0])

    def __init__(self, Nplayers, possibleActions=None) -> None:
        self.NPlayers = Nplayers
        # Check possible action Shape
        if (possibleActions != None):
            self.setPossibleActions(possibleActions)
        pass

    def setPossibleActions(self, possibleActions: np.ndarray) -> None:
        self.possibleActions = possibleActions

        # Fill the transition matrix
        profiles = [TransitionProfile({}) for i in range(
            np.prod(possibleActions))]

        self.transitionMatrix = np.array(profiles).reshape(possibleActions)

        self.payoffMatrix = np.zeros(
            tuple(possibleActions) + tuple([self.NPlayers]), dtype=np.float)
        pass

    def setTransition(self, actionProfile: np.ndarray, nextGame, probaility: float) -> None:
        self.transitionMatrix[actionProfile].setTransition(
            nextGame, probaility)
        pass

    def getTransition(self, actionProfile: np.ndarray) -> TransitionProfile:
        return self.transitionMatrix[actionProfile]

    def setPayoff(self, actionProfile: np.ndarray, payoff: np.ndarray) -> None:
        self.payoffMatrix[actionProfile] = payoff
        pass

    def getPayoff(self, actionProfile: np.ndarray) -> np.ndarray:
        return self.payoffMatrix[actionProfile]

    def __str__(self) -> str:
        return f"Game: {self.possibleActions} \n NPlayers: {self.NPlayers} \n Transition: {self.transitionMatrix} \n Payoff: {self.payoffMatrix}"


class Environment:
    Games: np.ndarray
    NPlayers: int = 2
    CurrentGame: Type[Game]

    def __init__(self, NGames: int = 2, NPlayers: int = 2) -> None:
        self.Games = np.array([Game(Nplayers=NPlayers) for i in range(NGames)])
        pass

    def getGames(self) -> np.ndarray:
        return self.Games

    def getGame(self, index: int) -> Type[Game]:
        return self.Games[index]

    def setCurrentGame(self, game: Type[Game]) -> None:
        self.CurrentGame = game
        pass

    def reward(self, actionProfile) -> float:
        return self.CurrentGame.getPayoff(actionProfile)

    def transitionProfile(self, actionProfile) -> TransitionProfile:
        return self.CurrentGame.getTransition(actionProfile)

    def performAction(self, actionProfile) -> None:
        reward = self.reward(actionProfile)
        self.CurrentGame = self.transitionProfile(
            actionProfile).sampleTransition()
        return reward


if __name__ == "__main__":
    game1 = Game(np.array([1, 2, 2]))
    game2 = Game(np.array([1]))

    print(game1)
    print(game2)

    transitionProfile = TransitionProfile({})
    transitionProfile.setTransition(game1, 0.5)
    transitionProfile.setTransition(game2, 0.5)

    print(transitionProfile)
    pass
