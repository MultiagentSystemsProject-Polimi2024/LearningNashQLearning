import numpy as np
from typing import Type, NewType


class TransitionProfile:
    def __init__(self, transitions: dict) -> None:
        self.transitions = transitions

    def setTransition(self, nextGame, probability: float) -> None:
        self.transitions[nextGame] = probability

    def getTransitions(self) -> tuple:
        return self.transitions.keys(), self.transitions.values()

    def getTransitionsDict(self) -> dict:
        return self.transitions

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
    def __init__(self, Nplayers, possibleActions=None) -> None:
        self.NPlayers = Nplayers
        self.transitionMatrix = None
        self.observers = []
        self.payoffMatrix = None
        self.possibleActions = np.array([])
        # Check possible action Shape
        if (possibleActions != None):
            self.setPossibleActions(possibleActions)
        pass

    def setPossibleActions(self, possibleActions: np.ndarray) -> None:
        # Padding the old possible actions with 1 to match the new possible actions
        self.possibleActions = np.pad(self.possibleActions, (0, max(0, len(possibleActions) - len(self.possibleActions))),
                                      mode='constant', constant_values=1)

        # Create a new transition matrix based on the new possible actions
        newTransitionMatrix = np.array([TransitionProfile({}) for i in range(
            np.prod(possibleActions))]).reshape(possibleActions)

        # Copy the old transition matrix to the new transition matrix
        if self.transitionMatrix is not None:
            slicingTuple = tuple([slice(0, min(
                possibleActions[i], self.possibleActions[i]), 1) for i in range(len(possibleActions))])

            newTransitionMatrix[slicingTuple] = self.transitionMatrix[slicingTuple]

        self.transitionMatrix = newTransitionMatrix

        # Create a new payoff matrix based on the new possible actions
        newPayoffMatrix = np.zeros(
            tuple(possibleActions) + tuple([self.NPlayers]), dtype=np.float)

        # Copy the old payoff matrix to the new payoff matrix
        if self.payoffMatrix is not None:
            slicingTuple = tuple([slice(0, min(
                possibleActions[i], self.possibleActions[i]), 1) for i in range(len(possibleActions))])
            newPayoffMatrix[slicingTuple] = self.payoffMatrix[slicingTuple]

        self.payoffMatrix = newPayoffMatrix

        # Set the new possible actions
        self.possibleActions = possibleActions

        self.notify()
        pass

    def getPossibleActions(self) -> np.ndarray:
        return self.possibleActions

    def setTransition(self, actionProfile: np.ndarray, nextGame, probaility: float) -> None:
        self.transitionMatrix[actionProfile].setTransition(
            nextGame, probaility)

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

        # trim the possible actions

        sliceTupleOld = tuple([slice(0, self.possibleActions[i], 1)
                               for i in range(len(self.possibleActions))])

        self.possibleActions = [
            self.possibleActions[i] for i in range(np.min([len(self.possibleActions), NPlayers]))
        ] + [1] * np.max([0, NPlayers - len(self.possibleActions)])

        sliceTupleNew = tuple([slice(0, self.possibleActions[i], 1)
                               for i in range(NPlayers)])

        playerSliceTuple = tuple([slice(0, min(NPlayers, self.NPlayers), 1)])

        self.NPlayers = NPlayers

        newTransitionMatrix = np.array([TransitionProfile({}) for i in range(
            np.prod(self.possibleActions))]).reshape(self.possibleActions)

        newTransitionMatrix[sliceTupleNew] = self.transitionMatrix[sliceTupleOld]

        self.transitionMatrix = newTransitionMatrix

        newPayoffMatrix = np.zeros(
            tuple(self.possibleActions) + tuple([NPlayers]), dtype=np.float)

        newPayoffMatrix[sliceTupleNew +
                        playerSliceTuple] = self.payoffMatrix[sliceTupleOld + playerSliceTuple]

        self.payoffMatrix = newPayoffMatrix

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


class EnvironmentObserver:
    def updateEnv(self, environment) -> None:
        pass


class Environment:
    def __init__(self, NGames: int = 2, NPlayers: int = 2) -> None:
        self.NPlayers = NPlayers
        possibleActions = tuple([1 for _ in range(NPlayers)])
        self.Games = np.array(
            [Game(Nplayers=NPlayers, possibleActions=possibleActions) for i in range(NGames)])
        self.CurrentGameIndex = 0
        self.observers = []
        pass

    def getGames(self) -> np.ndarray:
        return self.Games

    def getGame(self, index: int) -> Type[Game]:
        return self.Games[index]

    def getNGames(self) -> int:
        return len(self.Games)

    def setCurrentGame(self, gameIndex) -> None:
        self.CurrentGameIndex = gameIndex
        pass

    def getCurrentGame(self) -> Type[Game]:
        return self.Games[self.CurrentGameIndex]

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

    def setNPlayers(self, nPlayers: int = 2):
        self.NPlayers = nPlayers

        for g in self.Games:
            g.setNPlayers(nPlayers)

    def setNGames(self, nGames: int = 2, possibleActions: np.ndarray = None):
        if possibleActions is None:
            possibleActions = tuple([1 for _ in range(self.NPlayers)])
        for _ in range(max(0, nGames - len(self.Games))):
            self.Games = np.append(self.Games, Game(
                Nplayers=self.NPlayers, possibleActions=possibleActions))

        self.Games = self.Games[:nGames]

        self.CurrentGameIndex = min(self.CurrentGameIndex, nGames - 1)

        self.notify()

    def attach(self, observer: Type[EnvironmentObserver]) -> None:
        self.observers.append(observer)
        pass

    def detach(self, observer: Type[EnvironmentObserver]) -> None:
        self.observers.remove(observer)
        pass

    def notify(self) -> None:
        for observer in self.observers:
            observer.updateEnv(self)
        pass


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
