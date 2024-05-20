from Model.Environment import Environment as Env
import ipywidgets as widgets
import numpy as np

class PresetGames:
    def __init__(self, env: Env):
        self.env = env
        self.selected = 0
        self.presets = [x for x in range(4)]
        self.__setupActions = [self.__reset, self.__firstSetup, self.__secondSetup, self.__thirdSetup]
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
        self.env = Env()
    
    def __firstSetup(self):
        env = self.env
        env.setNPlayers(2)
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
        games[1].setPayoff(1, 1, 1, 1), np.array([-2,-2,-2,-2])