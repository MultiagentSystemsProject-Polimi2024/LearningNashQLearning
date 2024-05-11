import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
import networkx as nx
from netgraph import Graph
from time import sleep
from threading import Thread, Condition, Lock
import seaborn as sns
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
        self.qtables_dict = {}
        self.q_tables = {}
        self.rewards = []

        # window percentage for smoothing rewards
        self.windowPercentage = 0.1

        # The game number
        self.gameNum = 0

        nashQ.attach(self)

        # Create main title
        self.title = widgets.HTML(
            value="<h1>Training Display</h1>")

        # Create the subTitle
        self.subTitle1 = widgets.HTML(
            value="<h2>Training History</h2>")

        # Create window slider
        self.window_slider = widgets.FloatSlider(
            value=self.windowPercentage, min=0, max=1, step=0.001, description='Window size percentage:', continuous_update=False, readout=True, readout_format='.2f')
        self.window_slider.observe(
            lambda change: self.setWindow(change['new']), names='value')

        # Create the graph output widget
        self.graphOut = widgets.Output()

        # Create the slider widget
        self.slider = widgets.IntSlider(value=self.gameNum, min=0, max=0, step=1, description='Game num:',
                                        continuous_update=False, readout=True, readout_format='d')
        self.slider.observe(self.__on_value_change, names='value')

        self.nextButton = widgets.Button(
            description='', icon='arrow-right')
        self.nextButton.on_click(
            lambda x: self.slider.set_trait(name='value', value=self.slider.value + 1 if self.slider.value < self.slider.max else self.slider.max))

        self.prevButton = widgets.Button(
            description='', icon='arrow-left')
        self.prevButton.on_click(
            lambda x: self.slider.set_trait(name='value', value=self.slider.value - 1 if self.slider.value > 0 else 0))

        self.sliderBox = widgets.HBox(
            [self.prevButton, self.slider, self.nextButton])

        # Create the output widget for the plots
        self.plotOut = widgets.Output()

        # Current Game Field
        self.currentGame = widgets.IntText(
            0, description='Current Game:', disabled=True)

        # Create the action profile widget
        self.actionProfileWidget = widgets.HBox()

        self.actionProfileSubTitle = widgets.HTML(
            value="<h3>Current Action Profile</h3>")

        self.actionProfileBox = widgets.VBox(
            [self.actionProfileSubTitle, self.actionProfileWidget])

        # Create the payoff widget
        self.payoffWidget = widgets.HBox()

        self.payoffWidgetSubTitle = widgets.HTML(
            value="<h3>Current Payoff</h3>")

        self.payoffBox = widgets.VBox(
            [self.payoffWidgetSubTitle, self.payoffWidget])

        # Create the Q Table widget
        self.qTableWidget = widgets.Tab()

        self.qTableSubTitle = widgets.HTML(
            value="<h3>Current Q-Tables</h3>")

        self.qTableBox = widgets.VBox(
            [self.qTableSubTitle, self.qTableWidget])

        # Create the VBox containing all the widgets
        self.box = widgets.VBox(
            [self.title, self.subTitle1,  self.window_slider, self.plotOut, self.sliderBox, self.graphOut, self.currentGame, self.actionProfileBox, self.payoffBox, self.qTableBox])

        self.graph = graphClass.GraphClass()
        self.graph.create_graph(env)
        self.graph.current_state_set(0)

        # create figure and axes
        with self.graphOut:
            self.graphFig, self.graphAx = plt.subplots(figsize=(6, 3))
            self.graphFig.canvas.header_visible = False
            self.graphAx.set_title('')
            self.graphAx.get_xaxis().set_visible(False)
            self.graphAx.get_yaxis().set_visible(False)
            self.graph.plotGraph(self.graphAx)
            plt.tight_layout()
            plt.show()

        with self.plotOut:
            sns.set_theme()
            self.plotFig, self.plotAx = plt.subplots(figsize=(6, 3))
            self.plotFig.canvas.header_visible = False
            self.plotAx.set_title('Rewards during the training episodes')
            self.plotAx.set_xlim(0, 1000)
            plt.tight_layout()
            plt.show()

    def update(self, gamesHistory: History, rewards):
        print('Updating')
        self.history = gamesHistory
        self.rewards = np.array(rewards)[:, :, 0]
        self.slider.max = len(self.rewards[0]) - 1
        self.setActionProfileDisplay(self.history.get(0).get('action_profile'))
        self.setPayoffDisplay(self.history.get(0).get('payoff'))
        self.setQTableDisplay(self.__getQTables())
        self.__plot_rewards(self.rewards)

    def setActionProfileDisplay(self, actionProfile):
        self.actionProfileWidget.children = []
        for i, action in enumerate(actionProfile):
            self.actionProfileWidget.children += (
                widgets.Text(value=str(action), description='Player ' + str(i) + ':', disabled=True),)

    def updateActionProfileDisplay(self, actionProfile):
        for i, action in enumerate(actionProfile):
            self.actionProfileWidget.children[i].value = str(action)

    def setPayoffDisplay(self, payoff):
        self.payoffWidget.children = []
        for i, pay in enumerate(payoff):
            self.payoffWidget.children += (
                widgets.Text(value=str(pay), description='Player ' + str(i) + ':', disabled=True),)

    def updatePayoffDisplay(self, payoff):
        for i, pay in enumerate(payoff):
            self.payoffWidget.children[i].value = str(pay)

    def setQTableDisplay(self, qTables):
        self.qTableWidget.children = []
        for i, qTable in enumerate(qTables):
            playerMatrix = widgets.Tab()
            for j, game in enumerate(qTable):
                qMatrix = widgets.GridBox()
                for k, q in enumerate(game):
                    for l, qCouple in enumerate(q):
                        pair = widgets.HBox()
                        for qValue in qCouple:
                            pair.children += (
                                widgets.Text(value=str(qValue), description='', disabled=True,
                                             layout=widgets.Layout(width='60px')),)

                        qMatrix.children += (pair,)
                qMatrix.layout.grid_template_columns = 'repeat(' + str(
                    len(qTable[i][j])) + ', 200px)'
                qMatrix.layout.grid_template_rows = 'repeat(' + str(
                    len(qTable[i][j][0])) + ', 80px)'
                playerMatrix.children += (qMatrix,)
                playerMatrix.set_title(j, 'Game ' + str(j))
            self.qTableWidget.children += (playerMatrix,)
            self.qTableWidget.set_title(i, 'Player ' + str(i))

    def updateQTableDisplay(self, qTables):
        for i, qTable in enumerate(qTables):
            for j, game in enumerate(qTable):
                for k, q in enumerate(game):
                    for l, qCouple in enumerate(q):
                        for m, qValue in enumerate(qCouple):
                            self.qTableWidget.children[i].children[j].children[k*len(qCouple) + l].children[m].value = str(
                                qValue)

    def __plot_rewards(self, rewards):

        window = int(self.windowPercentage *
                     len(self.history.getHistory()))

        rewards_players, rewards_sum = self.__smooth_rewards(
            rewards, window)

        self.plotAx.clear()

        for (i, player) in enumerate(rewards_players):
            self.plotAx.plot(player, label='Player' + str(i))

        self.plotAx.plot(rewards_sum, label='Sum')
        self.plotAx.set_title(
            'Rewards during the training episodes - Smoothed with window size ' + str(window))
        self.plotAx.set_xlabel('Episodes')
        self.plotAx.set_ylabel('Reward')
        self.plotAx.legend()

        # plot the vertical line
        self.line = self.plotAx.axvline(
            x=self.gameNum, color='r', linestyle='--', label='Current Game')

    def __smooth_rewards(self, rewards, window=100):
        if window == 0:
            return rewards, np.sum(rewards, axis=0)

        rewardsPlayersSmooth = []
        for i in range(rewards.shape[0]):
            cumsum = np.cumsum(rewards[i, :])
            cumsum[window:] = cumsum[window:] - cumsum[:-window]

            cumsum = cumsum[window - 1:] / window

            # interpolate the cumsum array to mantain the same size
            cumsum = np.insert(cumsum, 0, np.zeros(window - 1))

            rewardsPlayersSmooth.append(cumsum)

        rewardsPlayersSmooth = np.array(rewardsPlayersSmooth)
        return rewardsPlayersSmooth, np.sum(rewardsPlayersSmooth, axis=0)

    def __plot_graph(self):
        self.graphAx.clear()
        current_state = self.history.get(self.gameNum).get('current_state')
        self.graph.current_state_set(current_state)
        self.graph.plotGraph(self.graphAx)

    def __getQTables(self):
        qTables = []
        for i in range(self.env.NPlayers):
            qTables.append(self.history.get(self.gameNum).get('Q' + str(i)))
        return qTables

    def __on_value_change(self, change):
        self.gameNum = int(change['new'])

        # clear the previous line
        # self.line.remove()
        self.line.set_xdata(self.gameNum)
        # self.line = self.plotAx.axvline(
        #     x=int(self.gameNum), color='r', linestyle='--')

        self.__plot_graph()

        # Update current game field
        self.currentGame.value = self.history.get(self.gameNum).get(
            'current_state')

        self.updateActionProfileDisplay(
            self.history.get(self.gameNum).get('action_profile'))

        self.updatePayoffDisplay(
            self.history.get(self.gameNum).get('payoff'))

        self.updateQTableDisplay(self.__getQTables())

        # with self.plotOut:
        #     self.plotOut.clear_output(True)
        #     for key in self.history.get(change['new']).keys():
        #         if key == 'current_state':
        #             continue

        #         print("Q-tables player" + key + ":\n")
        #         for q in self.history.get(change['new']).get(key):
        #             print(q)
        #             print("\n")

    def setWindow(self, windowPercentage: int):
        self.windowPercentage = windowPercentage
        if self.history is not None:
            self.__plot_rewards(self.rewards)

    def get_widget(self):
        return self.box
