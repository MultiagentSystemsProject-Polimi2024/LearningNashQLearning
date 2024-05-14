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
                    sleep(1)
                    self.envGraphDisplay.timer -= 1
                # self.envGraphDisplay.update_graph()


class EnvGraphDisplay(EnvironmentObserver):
    def __init__(self, env: Environment, timeBuffer=2):
        self.env = env
        env.attach(self)

        self.timeBuffer = timeBuffer
        self.timer = timeBuffer
        self.resetTimer()
        self.timerLock = Lock()
        self.timerCondition = Condition(self.timerLock)

        self.timerThread = CounterThread(self)
        self.timerThread.start()

        self.out = widgets.Output()
        self.button = widgets.Button(description='Update Graph')
        self.vBox = widgets.VBox([self.out, self.button])

        with self.out:
            fig, self.ax = plt.subplots()
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)

        self.button.on_click(lambda x: self.update_graph())

        self.updateEnv(env)
        self.update_graph()

    def resetTimer(self):
        self.timer = self.timeBuffer

    """"
    def updateEnv(self, env: Environment):
        if (not self.timerLock.locked()):
            with self.timerCondition:
                self.timerCondition.notify()
        self.resetTimer()
    """

    def updateEnv(self, env: Environment):
        self.env = env

    def countdown(self):
        while True:
            sleep(2)
            self.__execute_tasks()

    def update_graph(self):
        print("updating graph")
        print("Current games: ", self.env.getGames())
        for game in self.env.getGames():
            print("Game: ", game)
            for action in game.getAllActionProfiles():
                games, probs = game.getTransition(
                tuple(action)).getTransitions()
                print("Action: ", action)
                for g, p in zip(games, probs):
                    p = round(p, 3)
                    print("Transition: g, p=", g, p)
        with self.out:
            graph = graphClass.GraphClass()
            graph.create_graph(self.env)
            self.ax.cla()
            graph.plotGraph(self.ax)

    def get_widget(self):
        return self.vBox
