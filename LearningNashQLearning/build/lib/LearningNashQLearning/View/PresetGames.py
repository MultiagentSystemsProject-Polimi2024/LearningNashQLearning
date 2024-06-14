import ipywidgets as widgets
import numpy as np
from ..Model.Environment import Environment as Env


class PresetGames:
    def __init__(self, env: Env):
        self.env = env
        self.selected = 0

        self.__setupActions = [self.__reset, self.__firstSetup, self.__secondSetup,
                               self.__thirdSetup, self.__prisonSetup, self.__littleGridWorld]

        self.presets = [x for x in range(self.__setupActions.__len__())]

        self.widget = widgets.Dropdown(
            options=self.presets,
            description='Preset:',
            disabled=False,
        )
        self.widget.observe(self.on_change, names='value')

    def getWidget(self) -> widgets.Dropdown:
        return self.widget

    def on_change(self, change):
        self.selected = change['new']
        self.__setupActions[self.selected]()

    def __reset(self):
        env = self.env
        env.setNPlayers(2)
        env.setNGames(2)
        games = env.getGames()
        games[0].setPossibleActions(np.array([1, 1]))
        games[0].setTransition((0, 0), 1, 0)
        games[0].setPayoff((0, 0), np.array([0, 0]))
        games[0].setTransition((0, 0), 0, 0)
        games[0].setPayoff((0, 0), np.array([0, 0]))

        games[1].setPossibleActions(np.array([1, 1]))
        games[1].setTransition((0, 0), 1, 0)
        games[1].setPayoff((0, 0), np.array([0, 0]))
        games[1].setTransition((0, 0), 0, 0)
        games[1].setPayoff((0, 0), np.array([0, 0]))

    def __firstSetup(self):
        self.__reset()
        env = self.env
        env.setNPlayers(2)
        env.setNGames(2)
        games = env.getGames()

        games[0].setPossibleActions(np.array([2, 2]))

        games[0].setTransition((0, 0), 1, 1)
        games[0].setPayoff((0, 0), np.array([1, 1]))

        games[0].setTransition((1, 0), 0, 1)
        games[0].setPayoff((1, 0), np.array([0, 0]))

        games[0].setTransition((0, 1), 0, 1)
        games[0].setPayoff((0, 1), np.array([0, 0]))

        games[0].setTransition((1, 1), 0, 1)
        games[0].setPayoff((1, 1), np.array([-1, -1]))

        ###############################################

        games[1].setPossibleActions(np.array([2, 2]))

        games[1].setTransition((0, 0), 1, 1)
        games[1].setPayoff((0, 0), np.array([1, 1]))

        games[1].setTransition((1, 0), 0, 1)
        games[1].setPayoff((1, 0), np.array([0, 0]))

        games[1].setTransition((0, 1), 0, 1)
        games[1].setPayoff((0, 1), np.array([0, 0]))

        games[1].setTransition((1, 1), 0, 1)
        games[1].setPayoff((1, 1), np.array([-1, -1]))

    def __secondSetup(self):
        self.__reset()
        env = self.env
        env.setNPlayers(3)
        env.setNGames(2)
        games = env.getGames()

        ################ Game 0####################
        games[0].setPossibleActions(np.array([2, 2, 2]))

        games[0].setTransition((0, 0, 0), 1, 1)
        games[0].setPayoff((0, 0, 0), np.array([2, 2, 2]))

        games[0].setTransition((0, 0, 1), 0, 1)
        games[0].setPayoff((0, 0, 1), np.array([1, 1, 1]))

        games[0].setTransition((0, 1, 0), 0, 1)
        games[0].setPayoff((0, 1, 0), np.array([1, 1, 1]))

        games[0].setTransition((0, 1, 1), 0, 1)
        games[0].setPayoff((0, 1, 1), np.array([0, 0, 0]))

        games[0].setTransition((1, 0, 0), 0, 1)
        games[0].setPayoff((1, 0, 0), np.array([1, 1, 1]))

        games[0].setTransition((1, 0, 1), 0, 1)
        games[0].setPayoff((1, 0, 1), np.array([0, 0, 0]))

        games[0].setTransition((1, 1, 0), 0, 1)
        games[0].setPayoff((1, 1, 0), np.array([0, 0, 0]))

        games[0].setTransition((1, 1, 1), 0, 1)
        games[0].setPayoff((1, 1, 1), np.array([-1, -1, -1]))

        ###############################################

        games[1].setPossibleActions(np.array([2, 2, 2]))

        games[1].setTransition((0, 0, 0), 1, 1)
        games[1].setPayoff((0, 0, 0), np.array([2, 2, 2]))

        games[1].setTransition((0, 0, 1), 0, 1)
        games[1].setPayoff((0, 0, 1), np.array([1, 1, 1]))

        games[1].setTransition((0, 1, 0), 0, 1)
        games[1].setPayoff((0, 1, 0), np.array([1, 1, 1]))

        games[1].setTransition((0, 1, 1), 0, 1)
        games[1].setPayoff((0, 1, 1), np.array([0, 0, 0]))

        games[1].setTransition((1, 0, 0), 0, 1)
        games[1].setPayoff((1, 0, 0), np.array([1, 1, 1]))

        games[1].setTransition((1, 0, 1), 0, 1)
        games[1].setPayoff((1, 0, 1), np.array([0, 0, 0]))

        games[1].setTransition((1, 1, 0), 0, 1)
        games[1].setPayoff((1, 1, 0), np.array([0, 0, 0]))

        games[1].setTransition((1, 1, 1), 0, 1)
        games[1].setPayoff((1, 1, 1), np.array([-1, -1, -1]))

    def __thirdSetup(self):
        self.__reset()
        env = self.env
        env.setNPlayers(4)
        env.setNGames(2)
        games = env.getGames()

        ################ Game 0####################
        games[0].setPossibleActions(np.array([2, 2, 2, 2]))

        games[0].setTransition((0, 0, 0, 0), 1, 1)
        games[0].setPayoff((0, 0, 0, 0), np.array([2, 2, 2, 2]))

        games[0].setTransition((0, 0, 0, 1), 0, 1)
        games[0].setPayoff((0, 0, 0, 1), np.array([1, 1, 1, 1]))

        games[0].setTransition((0, 0, 1, 0), 0, 1)
        games[0].setPayoff((0, 0, 1, 0), np.array([1, 1, 1, 1]))

        games[0].setTransition((0, 0, 1, 1), 0, 1)
        games[0].setPayoff((0, 0, 1, 1), np.array([0, 0, 0, 0]))

        games[0].setTransition((0, 1, 0, 0), 0, 1)
        games[0].setPayoff((0, 1, 0, 0), np.array([1, 1, 1, 1]))

        games[0].setTransition((0, 1, 0, 1), 0, 1)
        games[0].setPayoff((0, 1, 0, 1), np.array([0, 0, 0, 0]))

        games[0].setTransition((0, 1, 1, 0), 0, 1)
        games[0].setPayoff((0, 1, 1, 0), np.array([0, 0, 0, 0]))

        games[0].setTransition((0, 1, 1, 1), 0, 1)
        games[0].setPayoff((0, 1, 1, 1), np.array([-1, -1, -1, -1]))
        #
        games[0].setTransition((1, 0, 0, 0), 0, 1)
        games[0].setPayoff((1, 0, 0, 0), np.array([1, 1, 1, 1]))

        games[0].setTransition((1, 0, 0, 1), 0, 1)
        games[0].setPayoff((1, 0, 0, 1), np.array([0, 0, 0, 0]))

        games[0].setTransition((1, 0, 1, 0), 0, 1)
        games[0].setPayoff((1, 0, 1, 0), np.array([0, 0, 0, 0]))

        games[0].setTransition((1, 0, 1, 1), 0, 1)
        games[0].setPayoff((1, 0, 1, 1), np.array([-1, -1, -1, -1]))

        games[0].setTransition((1, 1, 0, 0), 0, 1)
        games[0].setPayoff((1, 1, 0, 0), np.array([0, 0, 0, 0]))

        games[0].setTransition((1, 1, 0, 1), 0, 1)
        games[0].setPayoff((1, 1, 0, 1), np.array([-1, -1, -1, -1]))

        games[0].setTransition((1, 1, 1, 0), 0, 1)
        games[0].setPayoff((1, 1, 1, 0), np.array([-1, -1, -1, -1]))

        games[0].setTransition((1, 1, 1, 1), 0, 1)
        games[0].setPayoff((1, 1, 1, 1), np.array([-2, -2, -2, -2]))

        ###############################################

        games[1].setPossibleActions(np.array([2, 2, 2, 2]))

        games[1].setTransition((0, 0, 0, 0), 1, 1)
        games[1].setPayoff((0, 0, 0, 0), np.array([2, 2, 2, 2]))

        games[1].setTransition((0, 0, 0, 1), 0, 1)
        games[1].setPayoff((0, 0, 0, 1), np.array([1, 1, 1, 1]))

        games[1].setTransition((0, 0, 1, 0), 0, 1)
        games[1].setPayoff((0, 0, 1, 0), np.array([1, 1, 1, 1]))

        games[1].setTransition((0, 0, 1, 1), 0, 1)
        games[1].setPayoff((0, 0, 1, 1), np.array([0, 0, 0, 0]))

        games[1].setTransition((0, 1, 0, 0), 0, 1)
        games[1].setPayoff((0, 1, 0, 0), np.array([1, 1, 1, 1]))

        games[1].setTransition((0, 1, 0, 1), 0, 1)
        games[1].setPayoff((0, 1, 0, 1), np.array([0, 0, 0, 0]))

        games[1].setTransition((0, 1, 1, 0), 0, 1)
        games[1].setPayoff((0, 1, 1, 0), np.array([0, 0, 0, 0]))

        games[1].setTransition((0, 1, 1, 1), 0, 1)
        games[1].setPayoff((0, 1, 1, 1), np.array([-1, -1, -1, -1]))
        #
        games[1].setTransition((1, 0, 0, 0), 0, 1)
        games[1].setPayoff((1, 0, 0, 0), np.array([1, 1, 1, 1]))

        games[1].setTransition((1, 0, 0, 1), 0, 1)
        games[1].setPayoff((1, 0, 0, 1), np.array([0, 0, 0, 0]))

        games[1].setTransition((1, 0, 1, 0), 0, 1)
        games[1].setPayoff((1, 0, 1, 0), np.array([0, 0, 0, 0]))

        games[1].setTransition((1, 0, 1, 1), 0, 1)
        games[1].setPayoff((1, 0, 1, 1), np.array([-1, -1, -1, -1]))

        games[1].setTransition((1, 1, 0, 0), 0, 1)
        games[1].setPayoff((1, 1, 0, 0), np.array([0, 0, 0, 0]))

        games[1].setTransition((1, 1, 0, 1), 0, 1)
        games[1].setPayoff((1, 1, 0, 1), np.array([-1, -1, -1, -1]))

        games[1].setTransition((1, 1, 1, 0), 0, 1)
        games[1].setPayoff((1, 1, 1, 0), np.array([-1, -1, -1, -1]))

        games[1].setTransition((1, 1, 1, 1), 0, 1)
        games[1].setPayoff((1, 1, 1, 1), np.array([-2, -2, -2, -2]))

    def __prisonSetup(self):
        self.__reset()
        env = self.env
        env.setNPlayers(2)
        env.setNGames(4)
        games = env.getGames()

        ################ Game 0####################
        games[0].setPossibleActions(np.array([2, 2]))

        games[0].setTransition((0, 0), 1, 1)
        games[0].setPayoff((0, 0), np.array([-5, -5]))

        games[0].setTransition((0, 1), 2, 1)
        games[0].setPayoff((0, 1), np.array([1, -10]))

        games[0].setTransition((1, 0), 3, 1)
        games[0].setPayoff((1, 0), np.array([-10, 1]))

        games[0].setTransition((1, 1), 1, 1)
        games[0].setPayoff((1, 1), np.array([-1, -1]))

        ################ Game 1####################
        games[1].setPossibleActions(np.array([3, 3]))

        games[1].setTransition((0, 0), 1, 1)
        games[1].setPayoff((0, 0), np.array([-1, -1]))

        games[1].setTransition((0, 1), 3, 1)
        games[1].setPayoff((0, 1), np.array([-2, 1]))

        games[1].setTransition((0, 2), 2, 1)
        games[1].setPayoff((0, 2), np.array([1, -2]))

        games[1].setTransition((1, 0), 2, 1)
        games[1].setPayoff((1, 0), np.array([1, -2]))

        games[1].setTransition((1, 1), 1, 1)
        games[1].setPayoff((1, 1), np.array([-1, -1]))

        games[1].setTransition((1, 2), 3, 1)
        games[1].setPayoff((1, 2), np.array([-2, 1]))

        games[1].setTransition((2, 0), 3, 1)
        games[1].setPayoff((2, 0), np.array([-2, 1]))

        games[1].setTransition((2, 1), 2, 1)
        games[1].setPayoff((2, 1), np.array([1, -2]))

        games[1].setTransition((2, 2), 1, 1)
        games[1].setPayoff((2, 2), np.array([-1, -1]))

        ################ Game 2####################
        games[2].setPossibleActions(np.array([2, 2]))

        games[2].setTransition((0, 0), 0, 1)
        games[2].setPayoff((0, 0), np.array([1, 1]))

        games[2].setTransition((0, 1), 1, 1)
        games[2].setPayoff((0, 1), np.array([-2, -1]))

        games[2].setTransition((1, 0), 2, 1)
        games[2].setPayoff((1, 0), np.array([0, -2]))

        games[2].setTransition((1, 1), 2, 1)
        games[2].setPayoff((1, 1), np.array([0, -1]))

        ################ Game 3####################
        games[3].setPossibleActions(np.array([2, 2]))

        games[3].setTransition((0, 0), 0, 1)
        games[3].setPayoff((0, 0), np.array([1, 1]))

        games[3].setTransition((0, 1), 3, 1)
        games[3].setPayoff((0, 1), np.array([-2, 0]))

        games[3].setTransition((1, 0), 1, 1)
        games[3].setPayoff((1, 0), np.array([-1, -2]))

        games[3].setTransition((1, 1), 3, 1)
        games[3].setPayoff((1, 1), np.array([-1, 0]))

    def packGridState(self, playerA: int, playerB: int):
        return 4*playerA + playerB

    def __littleGridWorld(self):
        self.__reset()
        env = self.env
        env.setNPlayers(2)
        env.setNGames(16)
        games = env.getGames()

        # Set the possible actions
        for game in games:
            game.setPossibleActions(np.array([3, 3]))

        # Set the transitions and rewards
        for playerA in range(4):
            for playerB in range(4):
                state = self.packGridState(playerA, playerB)
                for actionA in range(3):
                    for actionB in range(3):
                        rewardA = 0
                        rewardB = 0

                        # Compute next state
                        nextA = playerA
                        if (actionA == 0 and playerA == 1):
                            nextA = 3
                        elif (actionA == 1 and playerA < 2):
                            nextA = playerA + 1
                        elif (actionA == 2 and playerA == 3):
                            nextA = 1

                        nextB = playerB
                        if (actionB == 0 and playerB == 1):
                            nextB = 3
                        elif (actionB == 1 and 1 <= playerB <= 2):
                            nextB = playerB - 1
                        elif (actionB == 2 and playerB == 3):
                            nextB = 1

                        if (nextA == nextB):
                            rewardA = -2
                            rewardB = -2
                            nextA = playerA
                            nextB = playerB

                        nextState = self.packGridState(nextA, nextB)

                        # Set the transition
                        games[state].setTransition(
                            (actionA, actionB), nextState, 1)

                        if (playerA == 2):
                            rewardA = 5
                        elif (playerA == nextA):
                            rewardA = -1

                        if (playerB == 2):
                            rewardB = 5
                        elif (playerB == nextB):
                            rewardB = -1

                        games[state].setPayoff(
                            (actionA, actionB), np.array([rewardA, rewardB]))
