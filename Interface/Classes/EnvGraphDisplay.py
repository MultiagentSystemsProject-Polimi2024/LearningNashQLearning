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
                self.envGraphDisplay.update_graph()


class EnvGraphDisplay(EnvironmentObserver):
    def __init__(self, env: Environment, timeBuffer=1):
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

        with self.out:
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

    def update_graph(self):
        graph = graphClass.GraphClass()
        graph.create_graph(self.env)
        self.ax.clear()
        graph.plot(self.ax, self.out)

    def get_widget(self):
        return self.out
