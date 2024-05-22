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
        nextGames = list(nextGames)
        probabilities = list(probabilities)
        return np.random.choice(nextGames, p=probabilities)

    def setNGames(self, nGames: int) -> None:
        keys = list(self.transitions.keys())
        for game in keys:
            if game >= nGames:
                del self.transitions[game]

    def isEmpty(self) -> bool:
        return len(self.transitions) == 0

    def __str__(self) -> str:
        return str(self.transitions)


class GameObserver:
    def update(self, game) -> None:
        pass


class Game:
    def __init__(self, Nplayers, possibleActions=None, id: int = 0) -> None:
        self.id = id
        self.NPlayers = Nplayers
        self.transitionMatrix = None
        self.observers = []
        self.payoffMatrix = None
        self.possibleActions = np.array([])
        # Check possible action Shape
        if (possibleActions != None):
            self.setPossibleActions(possibleActions)
        pass

    def getAllActionProfiles(self) -> np.ndarray:
        return np.array(np.meshgrid(*[range(i) for i in self.possibleActions])).T.reshape(-1, self.NPlayers)

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
            tuple(possibleActions) + tuple([self.NPlayers]), dtype=float)

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

        if (NPlayers < self.NPlayers):
            self.possibleActions = self.possibleActions[:NPlayers]

            slicingTuple = tuple([slice(0, self.possibleActions[i]) for i in range(
                NPlayers)]) + tuple([0 for i in range(self.NPlayers - NPlayers)])

            self.transitionMatrix = self.transitionMatrix[slicingTuple]

            self.payoffMatrix = self.payoffMatrix[slicingTuple +
                                                  tuple([slice(0, NPlayers)])]

        elif (NPlayers > self.NPlayers):
            self.possibleActions = np.pad(
                self.possibleActions, (0, NPlayers - self.NPlayers), 'constant', constant_values=1)

            slicingTuple = tuple([slice(0, self.possibleActions[i]) for i in range(
                self.NPlayers)]) + tuple([0 for i in range(NPlayers - self.NPlayers)])

            newTransitionMatrix = np.array([TransitionProfile({}) for _ in range(
                np.prod(self.possibleActions))]).reshape(self.possibleActions)

            newTransitionMatrix[slicingTuple] = self.transitionMatrix

            self.transitionMatrix = newTransitionMatrix

            newPayoffMatrix = np.zeros(
                tuple(self.possibleActions) + tuple([NPlayers]))

            newPayoffMatrix[slicingTuple +
                            tuple([slice(0, self.NPlayers)])] = self.payoffMatrix

            self.payoffMatrix = newPayoffMatrix

        self.NPlayers = NPlayers

        self.notify()
        pass

    def getTransitionMatrixStr(self) -> str:
        linearized = np.reshape(self.transitionMatrix,
                                np.prod(self.possibleActions))
        return str([
            str(linearized[i]) for i in range(np.prod(self.possibleActions))
        ])

    def getTransitionMatrix(self) -> np.ndarray:
        return self.transitionMatrix

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

    def setNGames(self, nGames: int) -> None:
        linearizedTransitionMatrix = np.reshape(
            self.transitionMatrix, np.prod(self.possibleActions))

        for transitionProfile in linearizedTransitionMatrix:
            transitionProfile.setNGames(nGames)

    def __str__(self) -> str:
        return f"""Game: {self.possibleActions} \n PossibileActions:{self.possibleActions} \n NPlayers: {self.NPlayers} \n Transition: {self.getTransitionMatrixStr()} \n Payoff: {self.payoffMatrix}"""


class EnvironmentObserver:
    def updateEnv(self, environment) -> None:
        pass


class GamesNObserver:
    def updateGames(self) -> None:
        pass


class Environment(GameObserver):
    def __init__(self, NGames: int = 2, NPlayers: int = 2) -> None:
        self.NPlayers = NPlayers
        possibleActions = tuple([1 for _ in range(NPlayers)])
        self.Games = np.array(
            [Game(Nplayers=NPlayers, possibleActions=possibleActions, id=i) for i in range(NGames)])
        for g in self.Games:
            g.attach(self)
        self.CurrentGameIndex = 0
        self.observers = []
        self.gameObservers = []
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

    def getCurrentGameIndex(self) -> int:
        return self.CurrentGameIndex

    def getGameIndex(self, game: Game) -> int:
        return np.where(self.Games == game)[0][0]

    def reward(self, actionProfile) -> float:
        return self.getCurrentGame().getPayoff(actionProfile)

    def transitionProfile(self, actionProfile) -> TransitionProfile:
        return self.getCurrentGame().getTransition(actionProfile)

    def transitionProfile(self, actionProfile, game: Game) -> TransitionProfile:
        return self.game.getTransition(actionProfile)
    
    def performAction(self, actionProfile) -> None:
        reward = self.reward(actionProfile)
        self.setCurrentGame(self.transitionProfile(
            actionProfile).sampleTransition())
        return reward

    def getNGames(self) -> int:
        return len(self.Games)

    def setNPlayers(self, nPlayers: int = 2):
        self.NPlayers = nPlayers

        for g in self.Games:
            g.setNPlayers(nPlayers)
        self.notify()

    def setNGames(self, nGames: int = 2, possibleActions: np.ndarray = None):
        if possibleActions is None:
            possibleActions = tuple([1 for _ in range(self.NPlayers)])

        for game in self.Games:
            game.setNGames(nGames)

        for _ in range(max(0, nGames - len(self.Games))):
            newGame = Game(Nplayers=self.NPlayers,
                           possibleActions=possibleActions, id=len(self.Games))
            newGame.attach(self)
            self.Games = np.append(self.Games, newGame)

        self.Games = self.Games[:nGames]

        self.CurrentGameIndex = min(self.CurrentGameIndex, nGames - 1)

        self.notify()
        self.notifyGames()

    def update(self, game: Game) -> None:
        self.notify()
        pass

    def attachGameObserver(self, observer: Type[GamesNObserver]) -> None:
        self.gameObservers.append(observer)
        pass

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

    def notifyGames(self) -> None:
        for observer in self.gameObservers:
            observer.updateGames()
        pass

    def getNextState(self, actionProfile) -> Game:
        return self.getGame(self.transitionProfile(
            actionProfile).sampleTransition())

    def setNextState(self, nextState: Game) -> Game:
        old = self.CurrentGameIndex
        self.CurrentGameIndex = np.where(self.Games == nextState)[0][0]
        return self.getGame(old)


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
