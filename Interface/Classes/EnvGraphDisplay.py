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
    from Interface.Classes.GraphPlotter import PlotGraph
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
                self.envGraphDisplay.update_graph()


class EnvGraphDisplay(EnvironmentObserver):
    def __init__(self, env: Environment, timeBuffer=3):
        self.env = env
        env.attach(self)
        self.graph = nx.DiGraph()
        self.edge_labels = {}
        self.node_colors = {}
        self.taskStack = []

        self.timeBuffer = timeBuffer
        self.timer = timeBuffer
        self.resetTimer()
        self.timerLock = Lock()
        self.timerCondition = Condition(self.timerLock)

        self.timerThread = CounterThread(self)
        self.timerThread.start()

        self.out = widgets.Output()

        # with self.out:
        fig, self.ax = plt.subplots()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.updateEnv(env)

    def resetTimer(self):
        self.timer = self.timeBuffer

    def updateEnv(self, env: Environment):

        if (not self.timerLock.locked()):
            with self.timerCondition:
                self.timerCondition.notify()

        self.resetTimer()

    def countdown(self):
        while True:
            sleep(2)
            self.__execute_tasks()

    def __execute_tasks(self):
        if (len(self.taskStack) == 0):
            return

        task = self.taskStack.pop()
        self.taskStack.clear()
        executor = Thread(target=task[0], args=[task[1]])
        executor.start()

    def update_graph(self):
        for game in self.env.getGames():
            self.graph.add_node(game)

            # find all non empty indexes
            actionProfiles = game.getAllActionProfiles()

            for action in actionProfiles:
                games, probs = game.getTransition(
                    tuple(action)).getTransitions()

                for g, p in zip(games, probs):
                    self.graph.add_edge(game, g)
                    if self.edge_labels.get((game, g)) is None:
                        self.edge_labels[(game, g)] = str(
                            action) + ': ' + str(p)
                    else:
                        self.edge_labels[(game, g)] += '\n' + \
                            str(action) + ': ' + str(p)

        for node in list(self.graph.nodes):
            self.node_colors[node] = 'b'

        print(self.node_colors)

        PlotGraph(
            self.graph, self.ax, self.edge_labels, self.node_colors).plot(self.out)

    def get_widget(self):
        return self.out
