import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns
import numpy as np
import networkx as nx
from netgraph import Graph
import GraphPlotter
from time import sleep
from threading import Thread

%matplotlib widget

class FinalDisplay:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.edge_labels = {}
        self.node_colors = {}
        self.old_line = []
        self.qtables_dict = {}
        self.gameNum = 0
        self.q_tables = {}
        self.axs = []
        self.history = []

        # executor thread
        self.taskStack = []
        self.bufferTimer = Thread(target=self.__countdown)
        self.bufferTimer.start()

        # create output widgets
        self.out = widgets.Output()
        self.slider = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Game num:', continuous_update=False, readout=True, readout_format='d')
        self.slider.observe(self.__on_value_change, names='value')
        self.output_widget = widgets.Output()
        widgets.display(self.out, self.slider, self.output_widget)

        # create figure and axes
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 15), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        self.axs[0].get_xaxis().set_visible(False)
        self.axs[0].get_yaxis().set_visible(False)
        self.axs[1].get_xaxis().set_visible(True)
        self.axs[1].get_yaxis().set_visible(True)
        self.axs[1].set_xlim(0, 1000)
        self.fig.tight_layout(h_pad=4)
        plt.show()

    def __countdown(self):
        sleep(5)
        self.__execute_tasks()

    def __execute_tasks(self):
        tasks = self.taskStack.pop()
        self.taskStack.clear()
        for task in tasks:
            executor = Thread(target=task[0], args=task[1])
    
    def update(self, gamesHistory, rewards):
        self.history = gamesHistory
        self.taskStack.append([(self.__plot_graph, self.gameNum), (self.__plot_rewards, rewards)])
    
    def __plot_rewards(self, rewards):
        self.slider.max = len(self.history) - 1
        sns.set_theme(style="whitegrid")
        rewards_players, rewards_sum = self.__smooth_rewards(rewards)
        i = 0
        for player in rewards_players:
            self.axs[1].plot(player, label='Player' + str(i))
            i += 1
        self.axs[1].plot(rewards_sum, label='Sum')
        self.axs[1].set_title('Rewards')
        self.axs[1].set_xlabel('Episodes')
        self.axs[1].set_ylabel('Rewards')
        self.axs[1].legend()
    
    def __smooth_rewards(self, rewards, window=100):
        rewardsPlayersSmooth = []
        rewardsSumSmooth = []
        for i in range(rewards.shape[1] - 1):
            cumsum = np.cumsum(rewards[:, i])
            cumsum[window:] = cumsum[window:] - cumsum[:-window]
            rewardsPlayersSmooth.append(cumsum[window - 1:] / window)
            rewardsSumSmooth = np.add(rewardsSumSmooth, rewardsPlayersSmooth[i])
        return rewardsPlayersSmooth, rewardsSumSmooth
    
    def __plot_graph(self, game_num):
        current_state = self.history[game_num]['current_state']
        for state in self.node_colors:
            if state == current_state:
                self.node_colors[state] = 'r'
            else:
                self.node_colors[state] = 'b'
        GraphPlotter.PlotGraph(self.graph, self.edge_labels, self.node_colors, self.axs[0]).plot()

    def __on_value_change(self, change):
        # print vertical red line in all subplots at the selected game number
        with self.out:
            # remove previous vertical line
            if not self.old_line == []:
                for line in self.old_line:
                    line.remove()
                self.old_line.clear()
            for ax in [self.axs[1], self.axs[2], self.axs[3]]:
                line = ax.axvline(x=change['new'], color='r', linestyle='--')
                self.old_line.append(line)
                ax.plot()
        self.__plot_graph(change['new'])
        with self.output_widget:
            self.output_widget.clear_output(True)
            for key in self.history[change['new']].keys():
                print("Q-tables player" + key + ":\n")
                for q in self.history[change['new']][key]:
                    print(q)
                    print("\n")