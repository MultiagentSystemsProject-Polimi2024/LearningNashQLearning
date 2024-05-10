import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
import networkx as nx
from netgraph import Graph
from time import sleep
from threading import Thread, Condition, Lock
import sys
sys.path.append('../../')

if True:
    import Interface.Classes.GraphClass as graphClass
    from Model.NashQLearning import NashQLearning, NashQLearningObserver
    from Model.Environment import Environment
    from Model.History import History
    from Interface.Classes.EnvGraphDisplay import EnvGraphDisplay

class FinalDisplay(NashQLearningObserver):

    def __init__(self, nashQ: NashQLearning, env: Environment):
        self.env = env
        self.old_line = []
        self.qtables_dict = {}
        self.q_tables = {}
        self.axs = []
        self.rewards = []
        nashQ.attach(self)

        # create output widgets
        self.out = widgets.Output()
        self.slider = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Game num:',
                                        continuous_update=False, readout=True, readout_format='d')
        self.slider.observe(self.__on_value_change, names='value')
        self.output_widget = widgets.Output()
        self.box = widgets.VBox([self.out, self.slider, self.output_widget])

        self.graph = graphClass.GraphClass()
        self.graph.create_graph(env)
        self.graph.current_state_set(0)

        # create figure and axes
        with self.out:
            self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 15), gridspec_kw={
                                              'height_ratios': [2, 1]})
            self.axs[0].get_xaxis().set_visible(False)
            self.axs[0].get_yaxis().set_visible(False)
            self.axs[1].get_xaxis().set_visible(True)
            self.axs[1].get_yaxis().set_visible(True)
            self.axs[1].set_xlim(0, 1000)
            self.fig.tight_layout(h_pad=4)
            plt.show()
            self.graph.plot(self.axs[0], self.out)

    def update(self, gamesHistory: History, rewards):
        self.history = gamesHistory
        self.rewards = np.array(rewards)[:,:,0]
        self.__plot_rewards(self.rewards)

    def __plot_rewards(self, rewards):
        self.slider.max = len(self.history.getHistory()) - 1
        rewards_players, rewards_sum = self.__smooth_rewards(rewards)
        i = 0
        with self.out:
            self.axs[1].clear()
            for player in rewards_players:
                self.axs[1].plot(player, label='Player' + str(i))
                i += 1
            self.axs[1].plot(rewards_sum, label='Sum')
            self.axs[1].set_title('Rewards during the training episodes')
            self.axs[1].set_xlabel('Episodes')
            self.axs[1].set_ylabel('Reward')
            self.axs[1].legend()

    def __smooth_rewards(self, rewards, window=100):
        rewardsPlayersSmooth = []
        for i in range(rewards.shape[0]):
            cumsum = np.cumsum(rewards[i, :])
            cumsum[window:] = cumsum[window:] - cumsum[:-window]
            rewardsPlayersSmooth.append(cumsum[window - 1:] / window)

        rewardsPlayersSmooth = np.array(rewardsPlayersSmooth)
        return rewardsPlayersSmooth, np.sum(rewardsPlayersSmooth, axis=0)

    def __plot_graph(self, game_num):
        current_state = self.history.get(game_num).get('current_state')
        self.graph.current_state_set(current_state)
        with self.out:
            self.graph.plot(self.axs[0], self.out)

    def __on_value_change(self, change):
        # print vertical red line in all subplots at the selected game number
        with self.out:
            # remove previous vertical line
            if not self.old_line == []:
                for line in self.old_line:
                    line.remove()
                self.old_line.clear()

            line = self.axs[1].axvline(
                x=change['new'], color='r', linestyle='--')
            self.old_line.append(line)
            self.axs[1].plot()
            self.__plot_graph(change['new'])
        with self.output_widget:
            self.output_widget.clear_output(True)
            for key in self.history.get(change['new']).keys():
                if key == 'current_state':
                    continue

                print("Q-tables player" + key + ":\n")
                for q in self.history.get(change['new']).get(key):
                    print(q)
                    print("\n")

    def get_widget(self):
        return self.box
