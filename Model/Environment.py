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

    def getProbability(self, nextGame) -> float:
        return self.transitions.get(nextGame, 0.0)

    def sampleTransition(self):
        nextGames, probabilities = self.getTransitions()
        return np.random.choice(nextGames, p=probabilities)

    def __str__(self) -> str:
        return str(self.transitions)


class GameObserver:
    def update(self, game) -> None:
        pass


class Game:
    NPlayers: int = 1
    possibleActions: np.ndarray = np.array([1])
    transitionMatrix: np.ndarray
    payoffMatrix: np.ndarray = np.array([.0])

    observers: list = []

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

        self.notify()
        pass

    def setTransition(self, actionProfile: np.ndarray, nextGame, probaility: float) -> None:
        self.transitionMatrix[actionProfile].setTransition(
            nextGame, probaility)
        print(self.transitionMatrix[actionProfile])

        self.notify()
        pass

    def setTransitionProfile(self, actionProfile: np.ndarray, transitionProfile: TransitionProfile) -> None:
        self.transitionMatrix[actionProfile] = transitionProfile
        self.notify()
        pass

    def getTransition(self, actionProfile: np.ndarray) -> TransitionProfile:
        return self.transitionMatrix[actionProfile]

    def setPayoff(self, actionProfile: np.ndarray, payoff: np.ndarray) -> None:
        self.payoffMatrix[actionProfile] = payoff
        self.notify()
        pass

    def getPayoff(self, actionProfile: np.ndarray) -> np.ndarray:
        return self.payoffMatrix[actionProfile]

    def setNPlayers(self, NPlayers: int) -> None:
        self.NPlayers = NPlayers

        # trim the possible actions

        self.possibleActions = [
            self.possibleActions[i] for i in range(np.min([len(self.possibleActions), NPlayers]))
        ] + [1] * np.max([0, NPlayers - len(self.possibleActions)])

        self.transitionMatrix = np.array(
            [TransitionProfile({}) for i in range(np.prod(self.possibleActions))]).reshape(self.possibleActions)

        self.payoffMatrix = np.zeros(
            tuple(self.possibleActions) + tuple([self.NPlayers]), dtype=np.float)

        self.notify()
        pass

    def getTransitionMatrixStr(self) -> str:
        linearized = np.reshape(self.transitionMatrix,
                                np.prod(self.possibleActions))
        return str([
            str(linearized[i]) for i in range(np.prod(self.possibleActions))
        ])

    def attach(self, observer: Type[GameObserver]) -> None:
        self.observers.append(observer)
        pass

    def detach(self, observer: Type[GameObserver]) -> None:
        self.observers.remove(observer)
        pass

    def notify(self) -> None:
        for observer in self.observers:
            observer.update(self)
        pass

    def __str__(self) -> str:
        return f"""Game: {self.possibleActions} \n PossibileActions:{self.possibleActions} \n NPlayers: {self.NPlayers} \n Transition: {self.getTransitionMatrixStr()} \n Payoff: {self.payoffMatrix}"""


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

    def getNGames(self) -> int:
        return len(self.Games)


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
