# this class actually implements the Nash-Q Learning algorithm, by having the agents chose their actions, based on the current policy, and by updating the Q-tables, from the class QTable, associated to the players after every action
import numpy as np
import pygambit as pg
from .History import History
from .Environment import Environment, Game, GamesNObserver
import ipywidgets as widgets


class QTable:
    def __init__(self, environment: Environment) -> None:
        self.table = {}
        for game in environment.getGames():
            shape = [game.getPossibleActions()[player] for player in range(
                environment.NPlayers)]  # Number of actions for each player
            shape.append(environment.NPlayers)  # Number of players
            self.table[environment.getGameIndex(
                game)] = np.zeros(shape, dtype=object)

    def getQTable(self):
        """returns the Qtable"""
        return self.table

    def getQvalue(self, state: Game, player: int, actions):
        """gets the Qvalue for a given state, player and actions"""
        return self.table[state][actions, player]

    def updateQvalue(self, state: Game, player: int, actions, alfa: float, gamma: float, nextQval, reward):
        """updates the Qvalue for a given state, player and actions"""
        self.table[state][actions, player] = (
            1 - alfa) * self.table[state][actions, player] + alfa * (reward + gamma * nextQval)

    def convertToTable(self):
        """converts the Qtable to a table returning a np.array object"""
        converted = []
        for game in self.table.keys():
            converted.append(self.table[game].copy())

        return np.array(converted, dtype=object)


class Agent:
    def __init__(self, number: int, environment: Environment) -> None:
        self.QTable = QTable(environment)
        self.environment = environment
        self.number = number
        self.currentAction = None
        self.currentStrategy = None

    def setCurrentStrategy(self, strategy: np.array):
        """sets the current strategy of the agent"""
        self.currentStrategy = strategy

    def __getPossibleActions(self, state: Game):
        """gets the possible actions for the agent"""
        return state.getPossibleActions()[self.number]

    def getQtable(self, state: Game):
        """gets the Qtable for the agent"""
        return self.QTable.getQTable()[self.environment.getGameIndex(state)]

    def Qtable(self):
        """get the Qtable for the agent"""
        return self.QTable
    
    def QtableForHistory(self):
        """gets the Qtable for the agent"""
        return self.QTable.convertToTable()

    def chooseAction(self, epsilon: float):
        """chooses the action for the agent given an index of randomness"""
        self.currentAction = np.random.choice(self.__getPossibleActions(self.environment.getCurrentGame()), p=self.currentStrategy) if np.random.rand(
        ) > epsilon else np.random.choice(self.__getPossibleActions(self.environment.getCurrentGame()))


class NashQLearningObserver:
    def update(self, history: History, NashQRewards) -> None:
        pass


class NashQLearning:
    def __init__(self, environment: Environment):

        self.env = environment
        self.episodes = 1
        self.epsilon = 0.1
        self.alfa = 0.5
        self.gamma = 0.8
        self.decaying_epsilon = 1000
        self.pure_training_ep = 1000
        self.goal_state = None
        self.startingState = self.env.getCurrentGame()  # caution
        self.reset = False

        self.history = History()
        self.NashQRewards = []

        self.observers = []

        self.widget = NashQLearningWidgets(self)

        # self.env.attach(self)

    def nashQlearning(self, alfa, gamma, epsilon, pure_training_ep, decaying_epsilon, reset=False, goal_state=None, startingState=None):
        """NashQ learning algorithm for n players. Parameters:\n
            alfa: \n
            gamma\n
            epsilon\n
            pure_training_ep\n
            decaying_epsilon\n
            reset\n
            goal_state\n
            starting_state\n"""
        self.prepareFunctions()

        # reset the values of the loading bar
        self.widget.gamesLoadingBarNashQ.value = 0

        # initialize class variables
        n_players = self.env.NPlayers

        self.agents = [Agent(i, self.env) for i in range(n_players)]

        self.goal_state = goal_state

        self.__already_seen_equilibria = {}

        if startingState != None:
            self.startingState = startingState
            self.env.setCurrentGame(self.env.getGameIndex(startingState))
        else:
            self.startingState = self.env.getCurrentGame()

        # initialize values to display
        self.totalReward = [np.zeros(n_players) for _ in range(n_players)]
        self.diffs = [[]for _ in range(n_players)]
        self.NashQRewards = [[]for _ in range(n_players)]
        self.history = History()

        for t in range(self.episodes):
            history_element = History()

            alfa = alfa / \
                (t + 1 - pure_training_ep) if t >= pure_training_ep else alfa
            epsilon = epsilon / \
                (t + 1 - decaying_epsilon) if t >= decaying_epsilon else epsilon

            # choose action for every player
            for agent in self.agents:
                agent.chooseAction(epsilon)

            actionProfile = tuple(
                [agent.currentAction for agent in self.agents])

            # calculating next state and memorizing its
            r = self.env.reward(actionProfile)
            nextState = self.env.getNextState(actionProfile)
            currentState = self.env.getCurrentGame()

            policy = []
            for agent in self.agents:

                # compute the expected payoff for the next state
                next_NashEq = self.computeNashEq(nextState, agent.Qtable())

                next_qVal = self.getNextQVal(
                    agent.Qtable(), nextState, next_NashEq)

                # copy modified Qvalues
                oldQ = self.copyQTable(
                    agent.Qtable(), currentState, actionProfile)

                # update qTable
                self.updateQTable(agent.Qtable(), actionProfile,
                                  alfa, gamma, r, next_qVal)

                # memorize the new qTable
                history_element.add(('Q'+str(agent.number)),
                                    agent.QtableForHistory())  # .copy())

                # memorize the agent's policy
                newStrategy = self.computeNashEq(currentState, agent.Qtable())
                policy.append(newStrategy)

                # memorize the difference between the old and the new value in the qTable
                self.diffs[agent.number].append(self.diffQTable(
                    agent.getQtable(currentState), oldQ, actionProfile))

                # update the total reward of the player i
                self.totalReward[agent.number] += r
                # memorize the reward of the player i
                self.NashQRewards[agent.number].append(r[agent.number])

            # memorize the state
            history_element.add(
                'current_state', self.env.getCurrentGameIndex())
            # memorize the action profile
            history_element.add(
                'action_profile', actionProfile)
            # memorize the policy
            history_element.add('policy', policy)
            # memorize the payoff
            history_element.add('payoff', r)

            # add the history element to the history
            self.history.add(t, history_element)

            # notify the observers every 1000 episodes
            # if t % min(100, self.episodes) == 0:
            #     self.notify(self.history, self.NashQRewards)

            # update the state
            if (reset and self.goal_state != None and currentState == self.goal_state):
                nextState = self.startingState

            self.playMove(nextState)

            # update the loading bar
            self.widget.gamesLoadingBarNashQ.value += 1

        self.notify(self.history, self.NashQRewards)
        self.widget.notifyEnd()
        return self.totalReward, self.diffs, self.NashQRewards, self.history


    def prepareFunctions(self):
        """prepares the functions for the nash equilibria computation and the qTable update"""
        # decide strategy for nash equilibria computation
        if self.env.NPlayers == 2:
            self.computeNashEq = self.twoPlNashEq
            self.updateQTable = self.updateQTable2
        elif self.env.NPlayers == 3:
            self.computeNashEq = self.threePlNashEq
            self.updateQTable = self.updateQTable3
        elif self.env.NPlayers == 4:
            self.computeNashEq = self.fourPlNashEq
            self.updateQTable = self.updateQTable4
        else:
            Exception("The number of players must be 2, 3 or 4")

    def startLearning(self):
        """starts the learning process"""
        self.nashQlearning(self.alfa, self.gamma, self.epsilon, self.pure_training_ep,
                           self.decaying_epsilon, self.reset, self.goal_state, self.startingState)

    def setEpisodes(self, episodes):
        """setter for the number of players"""
        self.episodes = episodes["new"]

    def playMove(self, nextState: Game) -> None:
        """setter for the number of players"""
        self.env.setNextState(nextState)
        return

    def qTable_to_tuple(self, qTable: np.array):
        """converts the qTable to a tuple"""
        try:
            return tuple(self.qTable_to_tuple(subarray) for subarray in qTable)
        except TypeError:
            return qTable

    def getKey(self, state: Game, table: np.array):
        """gets the key for the qTable"""
        s = self.env.getGameIndex(state)
        t = self.qTable_to_tuple(table)
        return (s, t)

    def twoPlNashEq(self, state: Game, qTable: QTable):
        """computes the Nash Equilibrium for a 2 players game"""

        payoff_matrix = qTable.getQTable()[self.env.getGameIndex(state)]
        # if the equilibrium has already been computed return it
        if self.getKey(state, payoff_matrix) in self.__already_seen_equilibria.keys():
            return self.__already_seen_equilibria[self.getKey(state, payoff_matrix)]

        game = pg.Game.from_arrays(
            payoff_matrix[:, :, 0], payoff_matrix[:, :, 1], title=("gambe number"+str(state)))
        # compute the Nash Equilibrium
        eq = pg.nash.enummixed_solve(game).equilibria
        # normalize the equilibrium
        eq = eq[0].normalize()

        # convert the Nash Equilibrium to an array
        tmp = []
        for i in range(self.env.NPlayers):
            x = np.zeros(state.getPossibleActions()[i])
            tmp.append(x)

        i = 0
        for p in game.players:
            for j in range(state.getPossibleActions()[i]):
                # TODO la cazzimma sta qui!
                tmp[i][j] = (float(eq[p][str(j+1)]))
            i += 1

        e = np.array(tmp, dtype=object)

        self.__already_seen_equilibria[self.getKey(state, payoff_matrix)] = e

        return e

    def threePlNashEq(self, state: Game, qTable: QTable):
        """computes the Nash Equilibrium for a 3 players game"""
        payoff_matrix = qTable.getQTable()[self.env.getGameIndex(state)]
        # if the equilibrium has already been computed return it
        if self.getKey(state, payoff_matrix) in self.__already_seen_equilibria.keys():
            return self.__already_seen_equilibria[self.getKey(state, payoff_matrix)]

        game = pg.Game.from_arrays(payoff_matrix[:, :, :, 0], payoff_matrix[:, :, :, 1],
                                   payoff_matrix[:, :, :, 2], title=("gambe number"+str(state)))
        # compute the Nash Equilibrium
        eq = pg.nash.logit_solve(game).equilibria
        # normalize the equilibrium
        eq = eq[0].normalize()
        # convert the Nash Equilibrium to an array
        tmp = []
        for i in range(self.env.NPlayers):
            x = np.zeros(state.getPossibleActions()[i])
            tmp.append(x)

        i = 0
        for p in game.players:
            for j in range(state.getPossibleActions()[i]):
                tmp[i][j] = (float(eq[p][str(j+1)]))
            i += 1

        e = np.array(tmp, dtype=object)

        self.__already_seen_equilibria[self.getKey(state, payoff_matrix)] = e

        return e

    def fourPlNashEq(self, state: Game, qTable: QTable):
        """compute the Nash Equilibrium for a 4 players game"""
        payoff_matrix = qTable.getQTable()[self.env.getGameIndex(state)]
        if self.getKey(state, payoff_matrix) in self.__already_seen_equilibria.keys():
            return self.__already_seen_equilibria[self.getKey(state, payoff_matrix)]

        game = pg.Game.from_arrays(payoff_matrix[:, :, :, :, 0], payoff_matrix[:, :, :, :, 1],
                                   payoff_matrix[:, :, :, :, 2], payoff_matrix[:, :, :, :, 3], title=("gambe number"+str(state)))
        # compute the Nash Equilibrium
        eq = pg.nash.logit_solve(game).equilibria
        # normalize the equilibrium
        eq = eq[0].normalize()
        # convert the Nash Equilibrium to an array
        tmp = []
        for i in range(self.env.NPlayers):
            x = np.zeros(state.getPossibleActions()[i])
            tmp.append(x)

        i = 0
        for p in game.players:
            for j in range(state.getPossibleActions()[i]):
                tmp[i][j] = (float(eq[p][str(j+1)]))
            i += 1

        e = np.array(tmp, dtype=object)

        self.__already_seen_equilibria[self.getKey(state, payoff_matrix)] = e

        return e

    
    def reward(self, state, player_actions, reward_matrix):
        """Getting reward for a given state and actions.\n
        The arguments must be reward_matrix, state, player1_action, player2_action, player3_action(optional)"""
        if self.n_players == 2:
            return reward_matrix[state, player_actions[0], player_actions[1]]
        elif self.n_players == 3:
            return reward_matrix[state, player_actions[0], player_actions[1], player_actions[2]]
        elif self.n_players == 4:
            return reward_matrix[state, player_actions[0], player_actions[1], player_actions[2], player_actions[3]]
        else:
            Exception("The number of players must be 2, 3 or 4")

    def expectedPayoff(self, payoff_matrix, player_strategies):
        """Getting the expected payoff in the future state.\n
        The arguments must be player1_strategy, player2_strategy, payoff_matrix"""
        if self.env.NPlayers == 2:
            return np.dot(player_strategies[1], np.dot(player_strategies[0], payoff_matrix))
        elif self.env.NPlayers == 3:
            return np.dot(player_strategies[2], np.dot(player_strategies[0], np.dot(player_strategies[1], payoff_matrix)))
        elif self.env.NPlayers == 4:
            return np.dot(player_strategies[3], np.dot(player_strategies[0], np.dot(player_strategies[1], np.dot(player_strategies[2], payoff_matrix))))
        else:
            Exception("The number of players must be 2, 3 or 4")

    # getting the expected payoff in the future state for n players
    def getNextQVal(self, qTable: QTable, next_state: Game, strategies: np.array):
        qTable = qTable.getQTable()
        next_qVal = np.zeros(self.env.NPlayers)
        for x in range(self.env.NPlayers):
            if self.env.NPlayers == 2:
                next_qVal[x] = self.expectedPayoff(
                    qTable[self.env.getGameIndex(next_state)][:, :, x], strategies)
            elif self.env.NPlayers == 3:
                next_qVal[x] = self.expectedPayoff(
                    qTable[self.env.getGameIndex(next_state)][:, :, :, x], strategies)
            elif self.env.NPlayers == 4:
                next_qVal[x] = self.expectedPayoff(
                    qTable[self.env.getGameIndex(next_state)][:, :, :, :, x], strategies)
            else:
                Exception("The number of players must be 2, 3 or 4")
        return next_qVal

    def copyQTable(self, qTable: QTable, state: Game, actions):
        """Returns a copy of the qTable in the state \"state\""""
        qTable = qTable.getQTable()
        if self.env.NPlayers > 4:
            raise Exception("The number of players must be 2, 3 or 4")
        return qTable[self.env.getGameIndex(state)][tuple(actions)].copy()

    # state must be dereferenced earlier than the other dimensions, being the qtable a dictionary qtable[state][...]...

    def updateQTable2(self, qTable: QTable, actions: np.array, alfa: float, gamma: float, r: np.array, next_qVal: np.array):
        """updates the qTable for 2 players"""
        state = self.env.getCurrentGame()
        qTable = qTable.getQTable()
        for x in range(self.env.NPlayers):
            qTable[self.env.getGameIndex(state)][actions[0], actions[1], x] = (
                1 - alfa) * qTable[self.env.getGameIndex(state)][actions[0], actions[1], x] + alfa * (r[x] + gamma * next_qVal[x])
            
    def updateQTable3(self, qTable: QTable, actions: np.array, alfa: float, gamma: float, r: np.array, next_qVal: np.array):
        """updates the qTable for 3 players"""
        state = self.env.getCurrentGame()
        qTable = qTable.getQTable()
        for x in range(self.env.NPlayers):
            qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], x] = (
                1 - alfa) * qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], x] + alfa * (r[x] + gamma * next_qVal[x])

    def updateQTable4(self, qTable: QTable, actions: np.array, alfa: float, gamma: float, r: np.array, next_qVal: np.array):
        """update the qTable for 4 players"""
        state = self.env.getCurrentGame()
        qTable = qTable.getQTable()
        for x in range(self.env.NPlayers):
            qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], actions[3], x] = (
                1 - alfa) * qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], actions[3], x] + alfa * (r[x] + gamma * next_qVal[x])

    def diffQTable(self, newTable: np.array, oldTable: np.array, actions: np.array):
        """returns the difference between the two qTables"""
        if self.env.NPlayers > 4:
            raise Exception("The number of players must be 2, 3 or 4")
        return newTable[tuple(actions)] - oldTable

    def attach(self, observer: NashQLearningObserver):
        """attaches an observer to the NashQLearning"""
        self.observers.append(observer)
        if (self.history != None and self.NashQRewards != None and self.NashQRewards != []):
            observer.update(self.history, self.NashQRewards)

    def detach(self, observer: NashQLearningObserver):
        """detaches an observer from the NashQLearning"""
        self.observers.remove(observer)

    def notify(self, history: History, NashQRewards):
        """notifies the observers of the NashQLearning"""
        for observer in self.observers:
            observer.update(history, NashQRewards)

    def getDisplayable(self):
        """returns the widgets"""
        return self.widget.getDisplayable()

    def getWidget(self):
        """returns the widget"""
        return self.widget


class NashQLearningWidgets (GamesNObserver):
    def __init__(self, nashQLearning: NashQLearning):

        self.nashQlearning = nashQLearning
        # widgets
        # reset widget
        self.resetWidget = widgets.Checkbox(
            value=False,
            description='Reset on goal state',
            disabled=False,
            indent=False,
            layout=widgets.Layout(justify_content='center'),

        )

        self.resetWidget.observe(self.setReset, names='value')

        # pure training episodes widget
        self.pureTrainingEpWidget = widgets.IntText(
            value=self.nashQlearning.pure_training_ep,
            layout=widgets.Layout(width='50%'),
            description='Pure training episodes:',
            style={'description_width': 'initial'},
            disabled=False
        )

        self.pureTrainingEpWidget.observe(
            self.setPureTrainingEp, names='value')

        # decaying epsilon widget
        self.decayingEpsilonWidget = widgets.IntText(
            value=self.nashQlearning.decaying_epsilon,
            layout=widgets.Layout(width='50%'),
            description='Pure epsilon episodes:',
            style={'description_width': 'initial'},
            disabled=False
        )

        self.decayingEpsilonWidget.observe(
            self.setDecayingEpsilon, names='value')

        # gamma widget
        self.gammaWidget = widgets.FloatText(
            value=self.nashQlearning.gamma,
            description='Gamma:',
            disabled=False,
            min=0,
            max=1
        )

        self.gammaWidget.observe(self.setGamma, names='value')

        # alfa widget
        self.alfaWidget = widgets.FloatText(
            value=self.nashQlearning.alfa,
            description='Alfa:',
            disabled=False,
            min=0,
            max=1
        )

        self.alfaWidget.observe(self.setAlfa, names='value')

        # epsilon widget
        self.epsilonWidget = widgets.FloatText(
            value=self.nashQlearning.epsilon,
            description='Epsilon:',
            disabled=False,
            min=0,
            max=1
        )

        self.epsilonWidget.observe(self.setEpsilon, names='value')

        # episodes widget
        self.episodesWidget = widgets.IntText(
            value=self.nashQlearning.episodes,
            description='Episodes:',
            disabled=False,
            min=1
        )

        self.episodesWidget.observe(self.setEpisodes, names='value')

        # goal state widget
        self.goalStateWidget = widgets.Dropdown(
            options=[(str(i), i)
                     for i in range(len(self.nashQlearning.env.getGames()))],
            description="Goal state: ",
            value=0,
            disabled=False,
        )

        self.goalStateWidget.observe(self.setGoalState, names='value')

        # starting state widget
        self.startingStateWidget = widgets.Dropdown(
            options=[(str(i), i)
                     for i in range(len(self.nashQlearning.env.getGames()))],
            description="Start state: ",
            value=0,
            disabled=False,
        )

        self.startingStateWidget.observe(self.setStartingState, names='value')

        # start button
        self.startButton = widgets.Button(description="Train")
        self.startButton.on_click(self.start)

        # loading bar
        self.gamesLoadingBarNashQ = widgets.IntProgress(
            value=0,
            min=0,
            max=1,
            step=1,
            description='Training:',
            bar_style='info',
        )

        # self.text = widgets.HTML(value="Tick if you want to restart every time the goal is reached:")
        self.text = widgets.Label(
            "Tick if you want to restart every time the goal is reached:")
        self.endLabel = widgets.Label("")
        self.grid = widgets.GridBox(layout=widgets.Layout(
            width='100%',
            grid_template_columns='repeat(2, 1fr)',
            grid_template_rows='repeat(7, 1fr)',
            grid_gap='10px'
        ))
        self.grid.children = [self.episodesWidget, self.gammaWidget, self.epsilonWidget,
                              self.decayingEpsilonWidget, self.alfaWidget, self.pureTrainingEpWidget,
                              self.text, self.resetWidget, self.startingStateWidget, self.goalStateWidget,
                              self.startButton, self.gamesLoadingBarNashQ, self.endLabel]

        nashQLearning.env.attachGameObserver(self)

    def notifyEnd(self):
        self.endLabel.value = "Training completed"

    def start(self, b):
        """starts the learning process"""
        self.endLabel.value = ""
        if (self.verifyIfSWellSet()):
            self.nashQlearning.startLearning()

    def setEpsilon(self, epsilon: float):
        self.nashQlearning.epsilon = epsilon["new"]

    def setAlfa(self, alfa: float):
        self.nashQlearning.alfa = alfa["new"]

    def setGamma(self, gamma: float):
        self.nashQlearning.gamma = gamma["new"]

    def setDecayingEpsilon(self, decaying_epsilon: int):
        self.nashQlearning.decaying_epsilon = decaying_epsilon["new"]

    def setPureTrainingEp(self, pure_training_ep: int):
        self.nashQlearning.pure_training_ep = pure_training_ep["new"]

    def setReset(self, reset: bool):
        self.nashQlearning.reset = reset["new"]

    def setGoalState(self, index: int):
        self.nashQlearning.goal_state = self.nashQlearning.env.getGame(
            index["new"])

    def setStartingState(self, index: int):
        self.nashQlearning.startingState = self.nashQlearning.env.getGame(
            index["new"])

    def getDisplayable(self):
        return self.grid

    def updateGames(self):
        self.goalStateWidget.options = [(str(i), i) for i in range(
            len(self.nashQlearning.env.getGames()))]

        self.startingStateWidget.options = [
            (str(i), i) for i in range(len(self.nashQlearning.env.getGames()))]

    def setEpisodes(self, episodes):
        self.nashQlearning.episodes = episodes["new"]
        self.gamesLoadingBarNashQ.max = self.nashQlearning.episodes-1

    def verifyIfSWellSet(self):
        """verifies if the environment is well set for the learning process, if not it displays an error message and returns False, otherwise it returns True"""
        self.endLabel.value = "verifying the env"
        if self.nashQlearning.env.getGames().shape[0] == 0:
            self.endLabel.value = "No games set"
            return False
        if self.nashQlearning.env.NPlayers == 0:
            self.endLabel.value = "No players set"
            return False
        for game in self.nashQlearning.env.getGames():
            if (len(game.getPossibleActions()) == 0 or game.getAllActionProfiles().shape[0] == []):
                self.endLabel.value = "No possible actions set in game "\
                    + str(self.nashQlearning.env.getGameIndex(game))
                return False

            actionProfiles = game.getAllActionProfiles()

            found = False
            for action in actionProfiles:
                games, probs = game.getTransition(
                    tuple(action)).getTransitions()
                probs_list = list(probs)
                for prob in probs_list:
                    if float(prob) != 0:
                        found = True
                        break

            if not found:
                self.endLabel.value = "No transitions set in game "\
                    + str(self.nashQlearning.env.getGameIndex(game))
                return False
        self.endLabel.value = ""
        return True
