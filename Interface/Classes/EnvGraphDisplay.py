import numpy as np
import networkx as nx
from time import sleep
from threading import Thread, Condition, Lock
import matplotlib.pyplot as plt
from ipywidgets import widgets
import logging
import sys
sys.path.append('../../')

if True:
    import Interface.Classes.GraphClass as graphClass
    from Model.Environment import Environment, EnvironmentObserver, Game, TransitionProfile


class CounterThread(Thread):
    def __init__(self, envGraphDisplay):
        Thread.__init__(self)
        self.envGraphDisplay = envGraphDisplay

    def run(self):
        while True:
            with self.envGraphDisplay.timerCondition:
                self.envGraphDisplay.timerCondition.wait()
                while self.envGraphDisplay.timer > 0:
                    sleep(0.1)
                    self.envGraphDisplay.timer -= 0.1
                self.envGraphDisplay.update_graph()


class EnvGraphDisplay(EnvironmentObserver):
    def __init__(self, env: Environment, timeBuffer=2.0):
        self.env = env
        env.attach(self)

        self.timeBuffer = timeBuffer
        self.timer = timeBuffer
        self.resetTimer()
        self.timerLock = Lock()
        self.timerCondition = Condition(self.timerLock)

        self.graph: graphClass.GraphClass = graphClass.GraphClass()

        self.timerThread = CounterThread(self)
        self.timerThread.start()

        self.labelOptions = {
            'Transition Probabilities': self.setLabelsToTp,
            'Payoffs': self.setLabelsToPayoff
        }

        self.labelOptionsDropdown = widgets.Dropdown(
            options=self.labelOptions.keys(),
            value=list(self.labelOptions.keys())[0],
            description='Labels:',
            disabled=False,
        )

        self.labelOptionsDropdown.observe(
            lambda x: self.updateEnv(self.env), names='value')

        self.out = widgets.Output()
        self.box = widgets.VBox([self.labelOptionsDropdown, self.out])

        with self.out:
            fig, self.ax = plt.subplots()
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)

        self.updateEnv(env)
        self.update_graph()

    def resetTimer(self):
        self.timer = self.timeBuffer

    def updateEnv(self, env: Environment):
        self.env = env
        if (not self.timerLock.locked()):
            with self.timerCondition:
                self.timerCondition.notify()
        self.resetTimer()

    def setLabelsToTp(self):
        for gameId, game in enumerate(self.env.getGames()):
            for action in game.getAllActionProfiles():
                games, probs = game.getTransition(
                    tuple(action)).getTransitions()
                for g, p in zip(games, probs):
                    self.graph.setActionLabel(
                        gameId, g, tuple(action), f'{p:.2f}')
        # self.updateEnv(self.env)

    def setLabelsToPayoff(self):
        for gameId, game in enumerate(self.env.getGames()):
            for action in game.getAllActionProfiles():
                payoffs = game.getPayoff(tuple(action))
                games, probs = game.getTransition(
                    tuple(action)).getTransitions()
                for g, p in zip(games, probs):
                    label = ''
                    for i, payoff in enumerate(payoffs):
                        label += f'{payoff:.2f} '
                    self.graph.setActionLabel(
                        gameId, g, tuple(action), label)
        # self.updateEnv(self.env)

    def countdown(self):
        while True:
            sleep(2)
            self.__execute_tasks()

    def update_graph(self):
        self.graph = graphClass.GraphClass()
        self.graph.create_graph(self.env)
        self.labelOptions[self.labelOptionsDropdown.value]()
        self.ax.cla()
        self.graph.plotGraph(self.ax)

    def get_widget(self):
        return self.box
        self.graph = graphClass.GraphClass()
        self.graph.create_graph(self.env)
        self.ax.clear()
        self.labelOptions[self.labelOptionsDropdown.value]()
        self.graph.plotGraph(self.ax)
