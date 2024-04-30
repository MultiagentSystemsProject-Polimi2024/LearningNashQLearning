import numpy as np
import networkx as nx
from time import sleep
from threading import Thread
import matplotlib.pyplot as plt
import GraphPlotter
from ipywidgets import widgets

class EnvGraphDisplay:
    def __init__(self, ax):
        self.graph = nx.DiGraph
        self.ax = ax
        self.edge_labels = {}
        self.node_colors = {}
        self.taskStack = []
        self.bufferTimer = Thread(target=self.__countdown)
        self.bufferTimer.start()

        self.out = widgets.Output()
        with self.out:
            fig, ax = plt.subplots()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.show()
    
    def update(self, env):
        self.taskStack.append((self.__update_graph, env))
    
    def __countdown(self):
        sleep(2)
        self.__execute_tasks()
    
    def __execute_tasks(self):
        task = self.taskStack.pop()
        self.taskStack.clear()
        executor = Thread(target=task[0], args=task[1])

    def __update_graph(self, env):
        for game in env.games:
            for action in game.getPossibleActions:
                transition = game.getTransition(action)
                nextGame = transition[0]
                self.graph.add_edge(game, nextGame)
                if self.edge_labels.get((game, nextGame)) is None:
                    self.edge_labels[(game, nextGame)] = str(action) + ': ' + str(transition[1])
                else:
                    self.edge_labels[(game, nextGame)] += '\n' + str(action) + ': ' + str(transition[1])
        for node in self.graph.nodes:
            self.node_colors[node] = 'b'
        with self.out:
            GraphPlotter.PlotGraph(self.graph, self.edge_labels, self.node_colors, self.ax).plot()
    
    def get_widget(self):
        return self.out