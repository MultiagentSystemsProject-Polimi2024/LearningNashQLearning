import numpy as np
from time import sleep
from threading import Thread, Condition, Lock
import matplotlib.pyplot as plt
from ipywidgets import widgets

from .GraphClass import GraphClass
from ..Model.Environment import Environment, EnvironmentObserver


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

        self.graph: GraphClass = GraphClass()

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

        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.fig.canvas.layout.min_height = '400px'
            self.fig.tight_layout()
            self.ax.set_aspect('equal')
            self.ax.axis('off')

        self.box = widgets.VBox([self.labelOptionsDropdown, self.fig.canvas])

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
                    if (p > 0):
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
                    if (p > 0):
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
        self.graph = GraphClass()
        self.graph.create_graph(self.env)
        self.labelOptions[self.labelOptionsDropdown.value]()
        self.graph.plotGraph(self.ax, self.fig)

    def get_widget(self):
        return self.box
