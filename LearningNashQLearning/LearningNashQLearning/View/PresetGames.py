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
            
        ################Game 0####################
        games[0].setPossibleActions(np.array([2, 2]))

        games[0].setTransition((0, 0), 1, 1)
        games[0].setPayoff((0, 0), np.array([-5,-5]))

        games[0].setTransition((0, 1), 2, 1)
        games[0].setPayoff((0, 1), np.array([1,-10]))

        games[0].setTransition((1, 0), 3, 1)
        games[0].setPayoff((1, 0), np.array([-10,1]))

        games[0].setTransition((1, 1), 1, 1)
        games[0].setPayoff((1, 1), np.array([-1,-1]))

        ################Game 1####################
        games[1].setPossibleActions(np.array([3, 3]))

        games[1].setTransition((0, 0), 1, 1)
        games[1].setPayoff((0, 0), np.array([-1,-1]))

        games[1].setTransition((0, 1), 3, 1)
        games[1].setPayoff((0, 1), np.array([-2,1]))

        games[1].setTransition((0, 2), 2, 1)
        games[1].setPayoff((0, 2), np.array([1,-2]))

        games[1].setTransition((1, 0), 2, 1)
        games[1].setPayoff((1, 0), np.array([1,-2]))

        games[1].setTransition((1, 1), 1, 1)
        games[1].setPayoff((1, 1), np.array([-1,-1]))

        games[1].setTransition((1, 2), 3, 1)
        games[1].setPayoff((1, 2), np.array([-2,1]))

        games[1].setTransition((2, 0), 3, 1)
        games[1].setPayoff((2, 0), np.array([-2,1]))
        
        games[1].setTransition((2, 1), 2, 1)
        games[1].setPayoff((2, 1), np.array([1,-2]))

        games[1].setTransition((2, 2), 1, 1)
        games[1].setPayoff((2, 2), np.array([-1,-1]))

        ################Game 2####################
        games[2].setPossibleActions(np.array([2, 2]))

        games[2].setTransition((0, 0), 0, 1)
        games[2].setPayoff((0, 0), np.array([1,1]))

        games[2].setTransition((0, 1), 1, 1)
        games[2].setPayoff((0, 1), np.array([-2,-1]))

        games[2].setTransition((1, 0), 2, 1)
        games[2].setPayoff((1, 0), np.array([0,-2]))

        games[2].setTransition((1, 1), 2, 1)
        games[2].setPayoff((1, 1), np.array([0,-1]))

        ################Game 3####################
        games[3].setPossibleActions(np.array([2, 2]))

        games[3].setTransition((0, 0), 0, 1)
        games[3].setPayoff((0, 0), np.array([1,1]))

        games[3].setTransition((0, 1), 3, 1)
        games[3].setPayoff((0, 1), np.array([-2,0]))

        games[3].setTransition((1, 0), 1, 1)
        games[3].setPayoff((1, 0), np.array([-1,-2]))

        games[3].setTransition((1, 1), 3, 1)
        games[3].setPayoff((1, 1), np.array([-1,0]))

    def __littleGridWorld(self):
        self.__reset()
        env = self.env
        env.setNPlayers(2)
        env.setNGames(12)
        games = env.getGames()

        ################################Game 0####################################
        games[0].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[0].setTransition((0, 1), 2, 1)
        games[0].setPayoff((0, 1), np.array([-1,0]))
        games[0].setTransition((2, 1), 2, 1)
        games[0].setPayoff((2, 1), np.array([-1,0]))
        games[0].setTransition((1, 0), 3, 1)
        games[0].setPayoff((1, 0), np.array([0,-1]))
        games[0].setTransition((1, 2), 3, 1)
        games[0].setPayoff((1, 2), np.array([0,-1]))
        
        ###############Negative Payoff Transitions###############
        games[0].setTransition((1, 1), 0, 1)
        games[0].setPayoff((1, 1), np.array([-2,-2]))

        ###############Non Valuable Transitions###############
        games[0].setTransition((0, 0), 0, 1)
        games[0].setPayoff((0, 0), np.array([-1,-1]))
        games[0].setTransition((0, 2), 0, 1)
        games[0].setPayoff((0, 2), np.array([-1,-1]))
        games[0].setTransition((2, 0), 0, 1)
        games[0].setPayoff((2, 0), np.array([-1,-1]))
        games[0].setTransition((2, 2), 0, 1)
        games[0].setPayoff((2, 2), np.array([-1,-1]))

        ################################Game 1####################################

        games[1].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[1].setTransition((0, 2), 3, 1)
        games[1].setPayoff((0, 2), np.array([-1,0]))
        games[1].setTransition((2, 2), 3, 1)
        games[1].setPayoff((2, 2), np.array([-1,0]))
        games[1].setTransition((1, 0), 4, 1)
        games[1].setPayoff((1, 0), np.array([0,-1]))
        games[1].setTransition((1, 1), 4, 1)
        games[1].setPayoff((1, 1), np.array([0,-1]))
        ###############Negative Payoff Transitions###############
        games[1].setTransition((1, 2), 1, 1)
        games[1].setPayoff((1, 2), np.array([-2,-2]))
        ###############Non Valuable Transitions###############
        games[1].setTransition((0, 0), 1, 1)
        games[1].setPayoff((0, 0), np.array([-1,-1]))
        games[1].setTransition((0, 1), 1, 1)
        games[1].setPayoff((0, 1), np.array([-1,-1]))
        games[1].setTransition((2, 0), 1, 1)
        games[1].setPayoff((2, 0), np.array([-1,-1]))
        games[1].setTransition((2, 1), 1, 1)
        games[1].setPayoff((2, 1), np.array([-1,-1]))

        ################################Game 2####################################
        games[2].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[2].setTransition((0, 0), 1, 1)
        games[2].setPayoff((0, 0), np.array([-1,0]))
        games[2].setTransition((2, 0), 1, 1)
        games[2].setPayoff((2, 0), np.array([-1,0]))
        games[2].setTransition((1, 0), 4, 1)
        games[2].setPayoff((1, 0), np.array([0,0]))
        ############Negative Payoff Transitions###############
        games[2].setTransition((1, 1), 2, 1)
        games[2].setPayoff((1, 1), np.array([-2,-2]))
        games[2].setTransition((0, 1), 2, 1)
        games[2].setPayoff((0, 1), np.array([-2,-2]))
        games[2].setTransition((1, 2), 2, 1)
        games[2].setPayoff((1, 2), np.array([-2,-2]))
        games[2].setTransition((2, 1), 2, 1)
        games[2].setPayoff((2, 1), np.array([-2,-2]))
        ############Non Valuable Transitions###############
        games[2].setTransition((0, 2), 2, 1)
        games[2].setPayoff((0, 2), np.array([-1,-1]))
        games[2].setTransition((2, 2), 2, 1)
        games[2].setPayoff((2, 2), np.array([-1,-1]))

        ################################Game 3####################################
        games[3].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[3].setTransition((0, 0), 9, 1)
        games[3].setPayoff((0, 0), np.array([0,-1]))
        games[3].setTransition((0, 2), 9, 1)
        games[3].setPayoff((0, 2), np.array([0,-1]))
        games[3].setTransition((0, 1), 10, 1)
        games[3].setPayoff((0, 1), np.array([0,0]))
        ############Negative Payoff Transitions###############
        games[3].setTransition((2, 0), 3, 1)
        games[3].setPayoff((2, 0), np.array([-1,-1]))
        games[3].setTransition((2, 2), 3, 1)
        games[3].setPayoff((2, 2), np.array([-1,-1]))
        ############Non Valuable Transitions###############
        games[3].setTransition((1, 1), 3, 1)
        games[3].setPayoff((1, 1), np.array([-2,-2]))
        games[3].setTransition((2, 1), 3, 1)
        games[3].setPayoff((2, 1), np.array([-2,-2]))
        games[3].setTransition((1, 2), 3, 1)
        games[3].setPayoff((1, 2), np.array([-2,-2]))
        games[3].setTransition((1, 0), 3, 1)
        games[3].setPayoff((1, 0), np.array([-2,-2]))

        ################################Game 4####################################
        games[4].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[4].setTransition((1, 2), 6, 1)
        games[4].setPayoff((1, 2), np.array([5,0]))
        games[4].setTransition((1, 0), 7, 1)
        games[4].setPayoff((1, 0), np.array([5,-1]))
        games[4].setTransition((1, 1), 7, 1)
        games[4].setPayoff((1, 1), np.array([5,-1]))
        ############Negative Payoff Transitions###############
        games[4].setTransition((0, 2), 4, 1)
        games[4].setPayoff((0, 2), np.array([-2,-2]))
        games[4].setTransition((0, 0), 4, 1)
        games[4].setPayoff((0, 0), np.array([-2,-2]))
        games[4].setTransition((0, 1), 4, 1)
        games[4].setPayoff((0, 1), np.array([-2,-2]))
        games[4].setTransition((2, 2), 4, 1)
        games[4].setPayoff((2, 2), np.array([-2,-2]))
        ############Non Valuable Transitions###############
        games[4].setTransition((2, 1), 4, 1)
        games[4].setPayoff((2, 1), np.array([-1,-1]))
        games[4].setTransition((2, 0), 4, 1)
        games[4].setPayoff((2, 0), np.array([-1,-1]))

        ################################Game 5####################################
        games[5].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[5].setTransition((1, 0), 8, 1)
        games[5].setPayoff((1, 0), np.array([5,5]))
        games[5].setTransition((1, 2), 8, 1)
        games[5].setPayoff((1, 2), np.array([5,5]))
        games[5].setTransition((0, 0), 11, 1)
        games[5].setPayoff((0, 0), np.array([0,5]))
        games[5].setTransition((0, 2), 11, 1)
        games[5].setPayoff((0, 2), np.array([0,5]))
        games[5].setTransition((0, 1), 11, 1)
        games[5].setPayoff((0, 1), np.array([0,5]))
        ############Negative Payoff Transitions###############
        
        ############Non Valuable Transitions###############
        games[5].setTransition((2, 0), 5, 1)
        games[5].setPayoff((2, 0), np.array([-1,5]))
        games[5].setTransition((2, 1), 5, 1)
        games[5].setPayoff((2, 1), np.array([-1,5]))
        games[5].setTransition((2, 2), 5, 1)
        games[5].setPayoff((2, 2), np.array([-1,5]))

        ################################Game 6####################################
        games[6].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[6].setTransition((0, 0), 7, 1)
        games[6].setPayoff((0, 0), np.array([5,0]))
        games[6].setTransition((1, 0), 7, 1)
        games[6].setPayoff((1, 0), np.array([5,0]))
        games[6].setTransition((2, 0), 7, 1)
        games[6].setPayoff((2, 0), np.array([5,0]))
        games[6].setTransition((0, 1), 8, 1)
        games[6].setPayoff((0, 1), np.array([5,5]))
        games[6].setTransition((1, 1), 8, 1)
        games[6].setPayoff((1, 1), np.array([5,5]))
        games[6].setTransition((0, 2), 8, 1)
        games[6].setPayoff((0, 2), np.array([5,5]))
        games[6].setTransition((2, 1), 8, 1)
        games[6].setPayoff((2, 1), np.array([5,5]))
        ############Non Valuable Transitions###############
        games[6].setTransition((1, 2), 6, 1)
        games[6].setPayoff((1, 2), np.array([5,0]))
        games[6].setTransition((2, 2), 6, 1)
        games[6].setPayoff((2, 2), np.array([5,0]))

        ################################Game 7####################################
        games[7].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[7].setTransition((0, 2), 6, 1)
        games[7].setPayoff((0, 2), np.array([5,0]))
        games[7].setTransition((1, 2), 6, 1)
        games[7].setPayoff((1, 2), np.array([5,0]))
        games[7].setTransition((2, 2), 6, 1)
        games[7].setPayoff((2, 2), np.array([5,0]))
        ############Negative Payoff Transitions###############

        ############Non Valuable Transitions###############
        games[7].setTransition((0, 0), 7, 1)
        games[7].setPayoff((0, 0), np.array([5,0]))
        games[7].setTransition((0, 1), 7, 1)
        games[7].setPayoff((0, 1), np.array([5,0]))
        games[7].setTransition((1, 0), 7, 1)
        games[7].setPayoff((1, 0), np.array([5,0]))
        games[7].setTransition((1, 1), 7, 1)
        games[7].setPayoff((1, 1), np.array([5,0]))
        games[7].setTransition((2, 0), 7, 1)
        games[7].setPayoff((2, 0), np.array([5,0]))
        games[7].setTransition((2, 1), 7, 1)
        games[7].setPayoff((2, 1), np.array([5,0]))

        ################################Game 8####################################
        games[8].setPossibleActions(np.array([3, 3]))

        ############Non Valuable Transitions###############
        games[8].setTransition((0, 0), 8, 1)
        games[8].setPayoff((0, 0), np.array([5,5]))
        games[8].setTransition((0, 1), 8, 1)
        games[8].setPayoff((0, 1), np.array([5,5]))
        games[8].setTransition((0, 2), 8, 1)
        games[8].setPayoff((0, 2), np.array([5,5]))
        games[8].setTransition((1, 0), 8, 1)
        games[8].setPayoff((1, 0), np.array([5,5]))
        games[8].setTransition((1, 1), 8, 1)
        games[8].setPayoff((1, 1), np.array([5,5]))
        games[8].setTransition((1, 2), 8, 1)
        games[8].setPayoff((1, 2), np.array([5,5]))
        games[8].setTransition((2, 0), 8, 1)
        games[8].setPayoff((2, 0), np.array([5,5]))
        games[8].setTransition((2, 1), 8, 1)
        games[8].setPayoff((2, 1), np.array([5,5]))
        games[8].setTransition((2, 2), 8, 1)
        games[8].setPayoff((2, 2), np.array([5,5]))

        ################################Game 9####################################
        games[9].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[9].setTransition((0, 1), 10, 1)
        games[9].setPayoff((0, 1), np.array([0,0]))
        games[9].setTransition((1, 1), 10, 1)
        games[9].setPayoff((1, 1), np.array([0,0]))
        games[9].setTransition((2, 0), 3, 1)
        games[9].setPayoff((2, 0), np.array([0,0]))
        games[9].setTransition((2, 2), 3, 1)
        games[9].setPayoff((2, 2), np.array([0,0]))
        ############Negative Payoff Transitions###############
        games[9].setTransition((2, 1), 9, 1)
        games[9].setPayoff((2, 1), np.array([-2,-2]))
        ############Non Valuable Transitions###############
        games[9].setTransition((0, 0), 9, 1)
        games[9].setPayoff((0, 0), np.array([-1,-1]))
        games[9].setTransition((0, 2), 9, 1)
        games[9].setPayoff((0, 2), np.array([-1,-1]))
        games[9].setTransition((1, 0), 9, 1)
        games[9].setPayoff((1, 0), np.array([-1,-1]))
        games[9].setTransition((1, 2), 9, 1)
        games[9].setPayoff((1, 2), np.array([-1,-1]))

        ################################Game 10####################################
        games[10].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[10].setTransition((0, 1), 11, 1)
        games[10].setPayoff((0, 1), np.array([-1,5]))
        games[10].setTransition((1, 1), 11, 1)
        games[10].setPayoff((1, 1), np.array([-1,5]))
        games[10].setTransition((2, 1), 11, 1)
        games[10].setPayoff((2, 1), np.array([0,5]))
        ############Negative Payoff Transitions###############
        games[10].setTransition((0, 0), 10, 1)
        games[10].setPayoff((0, 0), np.array([-2,-2]))
        games[10].setTransition((1, 0), 10, 1)
        games[10].setPayoff((1, 0), np.array([-2,-2]))
        games[10].setTransition((2, 0), 10, 1)
        games[10].setPayoff((2, 0), np.array([-2,-2]))
        games[10].setTransition((2, 2), 10, 1)
        games[10].setPayoff((2, 2), np.array([-2,-2]))
        ############Non Valuable Transitions###############
        games[10].setTransition((0, 2), 10, 1)
        games[10].setPayoff((0, 2), np.array([-1,-1]))
        games[10].setTransition((1, 2), 10, 1)
        games[10].setPayoff((1, 2), np.array([-1,-1]))

        ################################Game 11####################################
        games[11].setPossibleActions(np.array([3, 3]))

        ###############Valuable Transitions###############
        games[11].setTransition((2, 0), 5, 1)
        games[11].setPayoff((2, 0), np.array([0,5]))
        games[11].setTransition((2, 1), 5, 1)
        games[11].setPayoff((2, 1), np.array([0,5]))
        games[11].setTransition((2, 2), 5, 1)
        games[11].setPayoff((2, 2), np.array([0,5]))
        ############Negative Payoff Transitions###############

        ############Non Valuable Transitions###############
        games[11].setTransition((0, 0), 11, 1)
        games[11].setPayoff((0, 0), np.array([-1,5]))
        games[11].setTransition((0, 1), 11, 1)
        games[11].setPayoff((0, 1), np.array([-1,5]))
        games[11].setTransition((0, 2), 11, 1)
        games[11].setPayoff((0, 2), np.array([-1,5]))
        games[11].setTransition((1, 0), 11, 1)
        games[11].setPayoff((1, 0), np.array([-1,5]))
        games[11].setTransition((1, 1), 11, 1)
        games[11].setPayoff((1, 1), np.array([-1,5]))
        games[11].setTransition((1, 2), 11, 1)
        games[11].setPayoff((1, 2), np.array([-1,5]))






        




        
        
        
        








