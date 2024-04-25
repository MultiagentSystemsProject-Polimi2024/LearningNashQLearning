import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns
import numpy as np
import random
import networkx as nx
from netgraph import Graph

class FinalDisplay:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.edge_labels = {}
        self.node_labels = {}
        self.node_colors = []
        self.old_line = []
        self.qtables_dict = {}

        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 15), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        self.fig.tight_layout(h_pad=4)

        plt.show()

    def __plot_rewards(self, rewards):
        sns.set_theme(style="whitegrid")

        rewards_0, rewards_1, rewards_sum = self.__smooth_rewards(rewards)

        self.axs[1].plot(rewards_0, label='Player 0')
        self.axs[1].plot(rewards_1, label='Player 1')
        self.axs[1].plot(rewards_sum, label='Sum')
        self.axs[1].set_title('Rewards')
        self.axs[1].set_xlabel('Episodes')
        self.axs[1].set_ylabel('Rewards')
        self.axs[1].legend()

    def __smooth_rewards(self, rewards, window=100):
        # p0
        cumsum = np.cumsum(rewards[:, 0])
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        rewardsSmooth_0 = cumsum[window - 1:] / window

        # p1
        cumsum = np.cumsum(rewards[:, 1])
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        rewardsSmooth_1 = cumsum[window - 1:] / window

        rewardsSmooth_sum = rewardsSmooth_0 + rewardsSmooth_1

        return rewardsSmooth_0, rewardsSmooth_1, rewardsSmooth_sum

    def __plot_graph(self, G, node_colors, edge_labels, ax):
        pass

    def __update_display(self, qtables_dict, game_num, axs):
        current_state = qtables_dict[game_num]['current_state']
        for state in self.node_colors:
            if state == current_state:
                self.node_colors[state] = 'r'
            else:
                self.node_colors[state] = 'k'
        self.__plot_graph(self.graph, self.node_colors, self.edge_labels, axs[0])
        Q1 = qtables_dict[game_num]['Q1']
        Q2 = qtables_dict[game_num]['Q2']

    def on_value_change(self, change, out):
        global old_line
        # print vertical red line in all subplots at the selected game number
        with out:
            # remove previous vertical line
            if not old_line == []:
                for line in old_line:
                    line.remove()
                old_line.clear()
            for ax in [axs[1], axs[2], axs[3]]:
                line = ax.axvline(x=change['new'], color='r', linestyle='--')
                old_line.append(line)
                ax.plot()
        self.__update_display(q_table_dict, change['new'])
        with output_widget:
            output_widget.clear_output(True)
            print("Q-tables player 1:\n")
            for q in qtables_dict[change['new']]['Q1']:
                print(q)
                print("\n")
            print("Q-tables player 2:\n")
            for q in qtables_dict[change['new']]['Q2']:
                print(q)
                print("\n")