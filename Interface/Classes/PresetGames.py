from Model.Environment import Environment as Env
import ipywidgets as widgets
import numpy as np

class PresetGames:
    def __init__(self, env: Env):
        self.env = env
        self.selected = 0
        self.presets = [x for x in range(5)]
        self.__setupActions = [self.__reset, self.__firstSetup, self.__secondSetup,
                                self.__thirdSetup, self.__prisonSetup]
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
        env = self.env
        env.setNPlayers(2)
        env.setNGames(2)
        games = env.getGames()

        games[0].setPossibleActions(np.array([2, 2]))

        games[0].setTransition((0, 0), 1, 1)
        games[0].setPayoff((0, 0), np.array([1,1]))

        games[0].setTransition((1, 0), 0, 1)
        games[0].setPayoff((1, 0), np.array([0,0]))

        games[0].setTransition((0, 1), 0, 1)
        games[0].setPayoff((0, 1), np.array([0,0]))

        games[0].setTransition((1, 1), 0, 1)
        games[0].setPayoff((1, 1), np.array([-1,-1]))

        ###############################################

        games[1].setPossibleActions(np.array([2, 2]))
        
        games[1].setTransition((0, 0), 1, 1)
        games[1].setPayoff((0, 0), np.array([1,1]))

        games[1].setTransition((1, 0), 0, 1)
        games[1].setPayoff((1, 0), np.array([0,0]))

        games[1].setTransition((0, 1), 0, 1)
        games[1].setPayoff((0, 1), np.array([0,0]))

        games[1].setTransition((1, 1), 0, 1)
        games[1].setPayoff((1, 1), np.array([-1,-1]))

    def __secondSetup(self):
        env = self.env
        env.setNPlayers(3)
        env.setNGames(2)
        games = env.getGames()

        ################Game 0####################
        games[0].setPossibleActions(np.array([2, 2, 2]))

        games[0].setTransition((0, 0, 0), 1, 1)
        games[0].setPayoff((0, 0, 0), np.array([2,2,2]))

        games[0].setTransition((0, 0, 1), 0, 1)
        games[0].setPayoff((0, 0, 1), np.array([1,1,1]))

        games[0].setTransition((0, 1, 0), 0, 1)
        games[0].setPayoff((0, 1, 0), np.array([1,1,1]))

        games[0].setTransition((0, 1, 1), 0, 1)
        games[0].setPayoff((0, 1, 1), np.array([0,0,0]))

        games[0].setTransition((1, 0, 0), 0, 1)
        games[0].setPayoff((1, 0, 0), np.array([1,1,1]))

        games[0].setTransition((1, 0, 1), 0, 1)
        games[0].setPayoff((1, 0, 1), np.array([0,0,0]))

        games[0].setTransition((1, 1, 0), 0, 1)
        games[0].setPayoff((1, 1, 0), np.array([0,0,0]))

        games[0].setTransition((1, 1, 1), 0, 1)
        games[0].setPayoff((1, 1, 1), np.array([-1,-1,-1]))

        ###############################################

        games[1].setPossibleActions(np.array([2, 2, 2]))

        games[1].setTransition((0, 0, 0), 1, 1)
        games[1].setPayoff((0, 0, 0), np.array([2,2,2]))

        games[1].setTransition((0, 0, 1), 0, 1)
        games[1].setPayoff((0, 0, 1), np.array([1,1,1]))

        games[1].setTransition((0, 1, 0), 0, 1)
        games[1].setPayoff((0, 1, 0), np.array([1,1,1]))

        games[1].setTransition((0, 1, 1), 0, 1)
        games[1].setPayoff((0, 1, 1), np.array([0,0,0]))

        games[1].setTransition((1, 0, 0), 0, 1)
        games[1].setPayoff((1, 0, 0), np.array([1,1,1]))

        games[1].setTransition((1, 0, 1), 0, 1)
        games[1].setPayoff((1, 0, 1), np.array([0,0,0]))

        games[1].setTransition((1, 1, 0), 0, 1)
        games[1].setPayoff((1, 1, 0), np.array([0,0,0]))

        games[1].setTransition((1, 1, 1), 0, 1)
        games[1].setPayoff((1, 1, 1), np.array([-1,-1,-1]))

    def __thirdSetup(self):

        env = self.env
        env.setNPlayers(4)
        env.setNGames(2)
        games = env.getGames()

        ################Game 0####################
        games[0].setPossibleActions(np.array([2, 2, 2, 2]))

        games[0].setTransition((0, 0, 0, 0), 1, 1)
        games[0].setPayoff((0, 0, 0, 0), np.array([2,2,2,2]))

        games[0].setTransition((0, 0, 0, 1), 0, 1)
        games[0].setPayoff((0, 0, 0, 1), np.array([1,1,1,1]))

        games[0].setTransition((0, 0, 1, 0), 0, 1)
        games[0].setPayoff((0, 0, 1, 0), np.array([1,1,1,1]))

        games[0].setTransition((0, 0, 1, 1), 0, 1)
        games[0].setPayoff((0, 0, 1, 1), np.array([0,0,0,0]))

        games[0].setTransition((0, 1, 0, 0), 0, 1)
        games[0].setPayoff((0, 1, 0, 0), np.array([1,1,1,1]))

        games[0].setTransition((0, 1, 0, 1), 0, 1)
        games[0].setPayoff((0, 1, 0, 1), np.array([0,0,0,0]))

        games[0].setTransition((0, 1, 1, 0), 0, 1)
        games[0].setPayoff((0, 1, 1, 0), np.array([0,0,0,0]))

        games[0].setTransition((0, 1, 1, 1), 0, 1)
        games[0].setPayoff((0, 1, 1, 1), np.array([-1,-1,-1,-1]))
        #
        games[0].setTransition((1, 0, 0, 0), 0, 1)
        games[0].setPayoff((1, 0, 0, 0), np.array([1,1,1,1]))

        games[0].setTransition((1, 0, 0, 1), 0, 1)
        games[0].setPayoff((1, 0, 0, 1), np.array([0,0,0,0]))

        games[0].setTransition((1, 0, 1, 0), 0, 1)
        games[0].setPayoff((1, 0, 1, 0), np.array([0,0,0,0]))

        games[0].setTransition((1, 0, 1, 1), 0, 1)
        games[0].setPayoff((1, 0, 1, 1), np.array([-1,-1,-1,-1]))

        games[0].setTransition((1, 1, 0, 0), 0, 1)
        games[0].setPayoff((1, 1, 0, 0), np.array([0,0,0,0]))

        games[0].setTransition((1, 1, 0, 1), 0, 1)
        games[0].setPayoff((1, 1, 0, 1), np.array([-1,-1,-1,-1]))

        games[0].setTransition((1, 1, 1, 0), 0, 1)
        games[0].setPayoff((1, 1, 1, 0), np.array([-1,-1,-1,-1]))

        games[0].setTransition((1, 1, 1, 1), 0, 1)
        games[0].setPayoff((1, 1, 1, 1), np.array([-2,-2,-2,-2]))

        ###############################################

        games[1].setPossibleActions(np.array([2, 2, 2, 2]))

        games[1].setTransition((0, 0, 0, 0), 1, 1)
        games[1].setPayoff((0, 0, 0, 0), np.array([2,2,2,2]))

        games[1].setTransition((0, 0, 0, 1), 0, 1)
        games[1].setPayoff((0, 0, 0, 1), np.array([1,1,1,1]))

        games[1].setTransition((0, 0, 1, 0), 0, 1)
        games[1].setPayoff((0, 0, 1, 0), np.array([1,1,1,1]))

        games[1].setTransition((0, 0, 1, 1), 0, 1)
        games[1].setPayoff((0, 0, 1, 1), np.array([0,0,0,0]))

        games[1].setTransition((0, 1, 0, 0), 0, 1)
        games[1].setPayoff((0, 1, 0, 0), np.array([1,1,1,1]))

        games[1].setTransition((0, 1, 0, 1), 0, 1)
        games[1].setPayoff((0, 1, 0, 1), np.array([0,0,0,0]))

        games[1].setTransition((0, 1, 1, 0), 0, 1)
        games[1].setPayoff((0, 1, 1, 0), np.array([0,0,0,0]))

        games[1].setTransition((0, 1, 1, 1), 0, 1)
        games[1].setPayoff((0, 1, 1, 1), np.array([-1,-1,-1,-1]))
        #
        games[1].setTransition((1, 0, 0, 0), 0, 1)
        games[1].setPayoff((1, 0, 0, 0), np.array([1,1,1,1]))

        games[1].setTransition((1, 0, 0, 1), 0, 1)
        games[1].setPayoff((1, 0, 0, 1), np.array([0,0,0,0]))

        games[1].setTransition((1, 0, 1, 0), 0, 1)
        games[1].setPayoff((1, 0, 1, 0), np.array([0,0,0,0]))

        games[1].setTransition((1, 0, 1, 1), 0, 1)
        games[1].setPayoff((1, 0, 1, 1), np.array([-1,-1,-1,-1]))

        games[1].setTransition((1, 1, 0, 0), 0, 1)
        games[1].setPayoff((1, 1, 0, 0), np.array([0,0,0,0]))

        games[1].setTransition((1, 1, 0, 1), 0, 1)
        games[1].setPayoff((1, 1, 0, 1), np.array([-1,-1,-1,-1]))

        games[1].setTransition((1, 1, 1, 0), 0, 1)
        games[1].setPayoff((1, 1, 1, 0), np.array([-1,-1,-1,-1]))

        games[1].setTransition((1, 1, 1, 1), 0, 1)
        games[1].setPayoff((1, 1, 1, 1), np.array([-2,-2,-2,-2]))

    def __prisonSetup(self):
        env = self.env
        env.setNPlayers(2)
        env.setNGames(4)
        games = env.getGames()
            
        ################Game 0####################
        games[0].setPossibleActions(np.array([2, 2]))

        for i in range(2):
            for j in range(2):
                games[0].setTransition((i, j), 0, 0)
                games[0].setTransition((i, j), 1, 0)
                games[0].setTransition((i, j), 2, 0)
                games[0].setTransition((i, j), 3, 0)
    
        games[0].setTransition((0, 0), 0, 1)
        games[0].setPayoff((0, 0), np.array([5,5]))

        games[0].setTransition((0, 1), 2, 1)
        games[0].setPayoff((0, 1), np.array([-10,5]))

        games[0].setTransition((1, 0), 3, 1)
        games[0].setPayoff((1, 0), np.array([5,-10]))

        games[0].setTransition((1, 1), 1, 1)
        games[0].setPayoff((1, 1), np.array([-1,-1]))

        ################Game 1####################
        games[1].setPossibleActions(np.array([3, 3]))

        for i in range(3):
            for j in range(3):
                games[1].setTransition((i, j), 0, 0)
                games[1].setTransition((i, j), 1, 0)
                games[1].setTransition((i, j), 2, 0)
                games[1].setTransition((i, j), 3, 0)

        games[1].setTransition((0, 0), 1, 1)
        games[1].setPayoff((0, 0), np.array([-1,-1]))

        games[1].setTransition((0, 1), 2, 1)
        games[1].setPayoff((0, 1), np.array([-2,1]))

        games[1].setTransition((0, 2), 3, 1)
        games[1].setPayoff((0, 2), np.array([1,-2]))

        games[1].setTransition((1, 0), 3, 1)
        games[1].setPayoff((1, 0), np.array([1,-2]))

        games[1].setTransition((1, 1), 1, 1)
        games[1].setPayoff((1, 1), np.array([-1,-1]))

        games[1].setTransition((1, 2), 2, 1)
        games[1].setPayoff((1, 2), np.array([-2,1]))

        games[1].setTransition((2, 0), 2, 1)
        games[1].setPayoff((2, 0), np.array([-2,1]))
        
        games[1].setTransition((2, 1), 3, 1)
        games[1].setPayoff((2, 1), np.array([1,-2]))

        games[1].setTransition((2, 2), 1, 1)
        games[1].setPayoff((2, 2), np.array([-1,-1]))

        ################Game 2####################
        games[2].setPossibleActions(np.array([2, 2]))

        for i in range(2):
            for j in range(2):
                games[2].setTransition((i, j), 0, 0)
                games[2].setTransition((i, j), 1, 0)
                games[2].setTransition((i, j), 2, 0)
                games[2].setTransition((i, j), 3, 0)

        games[2].setTransition((0, 0), 0, 1)
        games[2].setPayoff((0, 0), np.array([1,1]))

        games[2].setTransition((0, 1), 2, 1)
        games[2].setPayoff((0, 1), np.array([-2,0]))

        games[2].setTransition((1, 0), 1, 1)
        games[2].setPayoff((1, 0), np.array([-2,-1]))

        games[2].setTransition((1, 1), 2, 1)
        games[2].setPayoff((1, 1), np.array([-1,0]))

        ################Game 3####################
        games[3].setPossibleActions(np.array([2, 2]))

        for i in range(2):
            for j in range(2):
                games[3].setTransition((i, j), 0, 0)
                games[3].setTransition((i, j), 1, 0)
                games[3].setTransition((i, j), 2, 0)
                games[3].setTransition((i, j), 3, 0)

        games[3].setTransition((0, 0), 0, 1)
        games[3].setPayoff((0, 0), np.array([1,1]))

        games[3].setTransition((0, 1), 1, 1)
        games[3].setPayoff((0, 1), np.array([-1,-2]))

        games[3].setTransition((1, 0), 3, 1)
        games[3].setPayoff((1, 0), np.array([0,-2]))

        games[3].setTransition((1, 1), 3, 1)
        games[3].setPayoff((1, 1), np.array([0,-1]))

