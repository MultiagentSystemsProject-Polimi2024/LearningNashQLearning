import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
import seaborn as sns
from .GraphClass import GraphClass
from ..Model.NashQLearning import NashQLearning, NashQLearningObserver
from ..Model.Environment import Environment, Game
from ..Model.History import History


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
                                        continuous_update=False, readout=True, readout_format='d', layout=widgets.Layout(width='50%', height='50px'))
        self.slider.observe(self.__on_value_change, names='value')

        self.playSlider = widgets.Play(
            value=self.gameNum, min=0, max=0, step=1, description='Game num:', interval=1000, continuous_update=False, disabled=False)

        widgets.jslink((self.playSlider, 'value'), (self.slider, 'value'))
        widgets.jslink((self.playSlider, 'max'), (self.slider, 'max'))

        self.nextButton = widgets.Button(
            description='', icon='arrow-right', layout=widgets.Layout(width='50px', height='50px'))
        self.nextButton.on_click(
            lambda x: self.slider.set_trait(name='value', value=self.slider.value + 1 if self.slider.value < self.slider.max else self.slider.max))

        self.prevButton = widgets.Button(
            description='', icon='arrow-left', layout=widgets.Layout(width='50px', height='50px'))
        self.prevButton.on_click(
            lambda x: self.slider.set_trait(name='value', value=self.slider.value - 1 if self.slider.value > 0 else 0))

        self.speedOptions = widgets.Dropdown(
            options=[1, 2, 5, 10, 20, 50, 100],
            value=1,
            description='Speed [game/sec]:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px', height='20px')
        )
        self.speedOptions.observe(
            lambda change: self.playSlider.set_trait(name='interval', value=1000/change['new']), names='value')

        self.sliderBox = widgets.VBox([
            widgets.HBox(
                [self.playSlider, self.slider, self.prevButton, self.nextButton]),
            self.speedOptions
        ])

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

        # Create the current Policy widget
        self.currentPolicyWidget = widgets.VBox()

        self.currentPolicySubTitle = widgets.HTML(
            value="<h3>Current Policy</h3>")

        self.currentPolicyBox = widgets.VBox(
            [self.currentPolicySubTitle, self.currentPolicyWidget])

        # Create the Q Table widget
        self.qTableWidget = widgets.Tab(value=0)

        self.qTableSubTitle = widgets.HTML(
            value="<h3>Current Q-Tables</h3>")

        self.qTableBox = widgets.VBox(
            [self.qTableSubTitle, self.qTableWidget])

        # Create the options for the graph labels
        self.labelOptions = {
            'NashQTable': self.__setLabelsToQTable,
            'NashQPolicy': self.__setLabelsToPolicy
        }

        self.graphLabelsOptions = widgets.Dropdown(
            options=self.labelOptions.keys(),
            value='NashQTable',
            description='Graph Labels:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px', height='20px')
        )

        self.graphLabelsOptions.observe(
            lambda x: self.__updateLabelsAndPlot(), names='value')

        self.targetPlayerOptions = widgets.Dropdown(
            options=[i for i in range(env.NPlayers)],
            value=0,
            description='Q tabel of player:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px', height='20px')
        )

        self.targetPlayerOptions.observe(
            lambda x: self.__updateLabelsAndPlot(), names='value')

        self.graphSettings = widgets.HBox(
            [self.targetPlayerOptions, self.graphLabelsOptions])

        self.graph = GraphClass()
        self.graph.create_graph(env)
        self.graph.current_state_set(0)

        # create figure and axes
        with plt.ioff():
            self.graphFig, self.graphAx = plt.subplots(figsize=(8, 8))
            self.graphFig.canvas.header_visible = False
            self.graphAx.set_title('')
            self.graphAx.get_xaxis().set_visible(False)
            self.graphAx.get_yaxis().set_visible(False)
            self.graph.plotGraph(self.graphAx, self.graphFig)
            # plt.tight_layout()

        with plt.ioff():
            sns.set_theme()
            self.plotFig, self.plotAx = plt.subplots(figsize=(6, 3))
            self.plotFig.canvas.header_visible = False
            self.plotAx.set_title('Rewards during the training episodes')
            self.plotAx.set_xlim(0, 1000)
            # plt.tight_layout()
            # plt.show()

        # Create the VBox containing all the widgets
        self.box = widgets.VBox(
            [self.title, self.subTitle1,  self.window_slider, self.plotFig.canvas, self.sliderBox, self.graphSettings, self.graphFig.canvas, self.currentGame, self.actionProfileBox, self.payoffBox, self.currentPolicyBox, self.qTableBox])

        nashQ.attach(self)

    def next(self):
        self.slider.set_trait(name='value', value=self.slider.value + 1)

    def update(self, gamesHistory: History, rewards):
        self.history = gamesHistory
        self.rewards = np.array(rewards)[:, :, 0]
        self.slider.max = len(self.rewards[0]) - 1
        self.setActionProfileDisplay(self.history.get(0).get('action_profile'))
        self.setPayoffDisplay(self.history.get(0).get('payoff'))
        self.setQTableDisplay(self.__getQTables())
        self.setCurrentPolicy(self.history.get(0).get(
            'policy')[self.targetPlayerOptions.value])
        self.__plot_rewards(self.rewards)

        self.targetPlayerOptions.options = [
            i for i in range(self.env.NPlayers)]

        self.graph.create_graph(self.env)
        self.__updateLabelsAndPlot()

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
        if self.env.NPlayers > 2:
            self.qTableWidget.children = [widgets.HTML(
                value='<h3>Q-Tables in matrix form are not supported for more than 2 players. Please refer to the graph display for this information</h3>')]
            return

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
                    len(qTable[j][k])) + ', 200px)'
                qMatrix.layout.grid_template_rows = 'repeat(' + str(
                    len(qTable[j][k][0])) + ', 80px)'
                playerMatrix.children += (qMatrix,)
                playerMatrix.set_title(j, 'Game ' + str(j))
            self.qTableWidget.children += (playerMatrix,)
            self.qTableWidget.set_title(i, 'Player ' + str(i))

    def updateQTableDisplay(self, qTables):
        if self.env.NPlayers > 2:
            return

        for i, qTable in enumerate(qTables):
            for j, game in enumerate(qTable):
                for k, q in enumerate(game):
                    for l, qCouple in enumerate(q):
                        for m, qValue in enumerate(qCouple):
                            self.qTableWidget.children[i].children[j].children[k*len(q) + l].children[m].value = str(
                                qValue)

    def setCurrentPolicy(self, policy):
        self.currentPolicyWidget.children = []
        for i, playerPolicy in enumerate(policy):
            playerPolicyWidget = widgets.HBox()
            label = widgets.Label(value='Player ' + str(i) + ':')
            for j, action in enumerate(playerPolicy):
                playerPolicyWidget.children += (
                    widgets.Text(value=str(round(action, 3)), description='Action ' + str(j) + ':', disabled=True),)
            self.currentPolicyWidget.children += (label,)
            self.currentPolicyWidget.children += (playerPolicyWidget,)

    def updateCurrentPolicy(self, policy):
        if len(policy) != len(self.currentPolicyWidget.children) / 2:
            self.setCurrentPolicy(policy)
            return

        for i, playerPolicy in enumerate(policy):
            if len(playerPolicy) != len(self.currentPolicyWidget.children[2*i + 1].children):
                self.setCurrentPolicy(policy)
                return

            for j, action in enumerate(playerPolicy):
                self.currentPolicyWidget.children[2*i + 1].children[j].value = str(
                    round(action, 3))

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

        self.plotAx.figure.canvas.draw()

    def __smooth_rewards(self, rewards, window=100):
        if window == 0:
            return rewards, np.sum(rewards, axis=0)

        rewardsPlayersSmooth = []
        for i in range(rewards.shape[0]):
            cumsum = np.cumsum(rewards[i, :])
            cumsum[window:] = cumsum[window:] - cumsum[:-window]

            cumsum = cumsum / window

            for i in range(1, window):
                cumsum[i] = cumsum[i] / i * window

            rewardsPlayersSmooth.append(cumsum)

        rewardsPlayersSmooth = np.array(rewardsPlayersSmooth)
        return rewardsPlayersSmooth, np.sum(rewardsPlayersSmooth, axis=0)

    def __plot_graph(self):
        current_state = self.history.get(self.gameNum).get('current_state')
        self.graph.current_state_set(current_state)

        if self.gameNum < len(self.history.getHistory()) - 1:
            next_state = self.history.get(self.gameNum+1).get('current_state')
            self.graph.setCurrentActionProfile(current_state, next_state)

        self.graph.plotGraph(self.graphAx, self.graphFig)

    def __getQTables(self):
        qTables = []
        for i in range(self.env.NPlayers):
            qTables.append(self.history.get(self.gameNum).get('Q' + str(i)))
        return qTables

    def __setLabelsToQTable(self):
        if self.gameNum != 0:
            pastQTable = self.history.get(self.gameNum - 1).get(
                'Q' + str(self.targetPlayerOptions.value))

        qTable = self.history.get(self.gameNum).get(
            'Q' + str(self.targetPlayerOptions.value))
        actionProfiles = np.ndenumerate(qTable.T[0].T)
        self.graph.clearActionLabels()

        for action, _ in actionProfiles:
            fromGame = action[0]
            toGames, ps = self.env.getGame(fromGame).getTransition(
                tuple(action[1:])).getTransitions()
            value = qTable[action[0]][action[1:]]

            if self.gameNum != 0:
                pastValue = pastQTable[action[0]][action[1:]]
                diff: np.array = value - pastValue

                valueStr = str([round(v, 2) for v in pastValue])

                if np.any(diff != 0):
                    valueStr += ' + ' + str([round(v, 2) for v in diff])
            else:
                valueStr = str([round(v, 2) for v in value])

            for toGame, p in zip(toGames, ps):
                if p > 0:
                    self.graph.setActionLabel(
                        fromGame, toGame, action[1:], valueStr)

    def __setLabelsToPolicy(self):
        self.graph.clearActionLabels()
        for gameIndex in range(self.env.getNGames()):
            localGameNum = self.gameNum
            while self.history.get(localGameNum).get('current_state') != gameIndex and localGameNum > 0:
                localGameNum -= 1

            if localGameNum == 0:
                continue

            policy = self.history.get(localGameNum).get(
                'policy')[self.targetPlayerOptions.value]

            # Convert the list of lists into a list of numpy arrays
            prob_arrays = [np.array(player_probs, dtype=np.float64)
                           for player_probs in policy]

            # Generate a grid of all possible action indices for each player
            grids = np.meshgrid(*[np.arange(len(player_probs))
                                for player_probs in policy], indexing='ij')

            # Initialize the joint probability array with ones
            joint_prob_shape = [len(player_probs) for player_probs in policy]
            joint_prob_array = np.ones(joint_prob_shape, dtype=np.float64)

            # Compute the joint probability by multiplying the probabilities of individual actions
            for i, grid in enumerate(grids):
                joint_prob_array *= prob_arrays[i][grid]

            game: Game = self.env.getGame(gameIndex)

            actionProfiles = game.getAllActionProfiles()

            for action in actionProfiles:
                games, probs = game.getTransition(
                    tuple(action)).getTransitions()
                for g, p in zip(games, probs):
                    if p > 0:
                        self.graph.setActionLabel(gameIndex, g, tuple(action), str(
                            round(joint_prob_array[tuple(action)], 2)))
        pass

    def __updateGraphLabels(self):
        self.labelOptions[self.graphLabelsOptions.value]()

    def __updateLabelsAndPlot(self):
        self.__updateGraphLabels()
        self.__plot_graph()

    def __on_value_change(self, change):
        self.gameNum = int(change['new'])

        # Update the vertical line
        self.line.set_xdata([self.gameNum])
        self.plotAx.figure.canvas.draw()

        self.__updateGraphLabels()
        self.__plot_graph()

        # Update current game field
        self.currentGame.value = self.history.get(self.gameNum).get(
            'current_state')

        self.updateActionProfileDisplay(
            self.history.get(self.gameNum).get('action_profile'))

        self.updatePayoffDisplay(
            self.history.get(self.gameNum).get('payoff'))

        self.updateQTableDisplay(self.__getQTables())

        self.updateCurrentPolicy(
            self.history.get(self.gameNum).get('policy')[self.targetPlayerOptions.value])

    def setWindow(self, windowPercentage: int):
        self.windowPercentage = windowPercentage
        if self.history is not None:
            self.__plot_rewards(self.rewards)

    def get_widget(self):
        return self.box
