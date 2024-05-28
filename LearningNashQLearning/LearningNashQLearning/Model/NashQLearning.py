# this class actually implements the Nash-Q Learning algorithm, by having the agents chose their actions, based on the current policy, and by updating the Q-tables, from the class QTable, associated to the players after every action
import numpy as np
import pygambit as pg
from .History import History
from .Environment import Environment, Game, GamesNObserver
from ..View.NashQLearningWidgets import NashQLearningWidgets

class QTable:
    def __init__(self, environment: Environment) -> None:
        self.table = {}
        for game in environment.getGames():
            shape = [game.getPossibleActions()[player] for player in range(
                environment.NPlayers)]  # Number of actions for each player
            shape.append(environment.NPlayers)  # Number of players
            self.table[environment.getGameIndex(
                game)] = np.zeros(shape, dtype=object)

    # get the Qtable
    def getQTable(self):
        return self.table

    # get the Qvalue for a given state, player and actions
    def getQvalue(self, state: Game, player: int, actions):
        return self.table[state][actions, player]

    # update the Qvalue for a given state, player and actions
    def updateQvalue(self, state: Game, player: int, actions, alfa: float, gamma: float, nextQval, reward):
        self.table[state][actions, player] = (
            1 - alfa) * self.table[state][actions, player] + alfa * (reward + gamma * nextQval)

    # convert the Qtable to a table
    def convertToTable(self):
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

    # set the current strategy of the agent
    def setCurrentStrategy(self, strategy: np.array):
        self.currentStrategy = strategy

    # get the possible actions for the agent
    def __getPossibleActions(self, state: Game):
        return state.getPossibleActions()[self.number]

    # get the Qtable for the agent
    def getQtable(self, state: Game):
        return self.QTable.getQTable()[self.environment.getGameIndex(state)]

   # get the Qtable for the agent 
    def Qtable(self):
        return self.QTable

    # get the Qtable for the agent
    def QtableForHistory(self):
        return self.QTable.convertToTable()

    # choose the action for the agent
    def chooseAction(self, epsilon: float):
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

    # NashQ learning algorithm for n players
    def nashQlearning(self, alfa, gamma, epsilon, pure_training_ep, decaying_epsilon, reset=False, goal_state=None, startingState=None):
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
                self.NashQRewards[agent.number].append(r)

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

    # prepare the functions for the nash equilibria computation and the qTable update
    def prepareFunctions(self):
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

    # start the learning process
    def startLearning(self):
        self.nashQlearning(self.alfa, self.gamma, self.epsilon, self.pure_training_ep,
                           self.decaying_epsilon, self.reset, self.goal_state, self.startingState)

    # setter for the number of players
    def setEpisodes(self, episodes):
        self.episodes = episodes["new"]

    # setter for the number of players
    def playMove(self, nextState: Game) -> None:
        self.env.setNextState(nextState)
        return

    # convert the qTable to a tuple
    def qTable_to_tuple(self, qTable: np.array):
        try:
            return tuple(self.qTable_to_tuple(subarray) for subarray in qTable)
        except TypeError:
            return qTable

    # get the key for the qTable
    def getKey(self, state: Game, table: np.array):
        s = self.env.getGameIndex(state)
        t = self.qTable_to_tuple(table)
        return (s, t)

    # compute the Nash Equilibrium for a 2 players game
    def twoPlNashEq(self, state: Game, qTable: QTable):

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

    # compute the Nash Equilibrium for a 3 players game
    def threePlNashEq(self, state: Game, qTable: QTable):
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

    # compute the Nash Equilibrium for a 4 players game
    def fourPlNashEq(self, state: Game, qTable: QTable):
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

    # getting reward for a given state and actions, the arguments must be
    # reward_matrix, state, player1_action, player2_action, player3_action(optional)
    def reward(self, state, player_actions, reward_matrix):
        if self.n_players == 2:
            return reward_matrix[state, player_actions[0], player_actions[1]]
        elif self.n_players == 3:
            return reward_matrix[state, player_actions[0], player_actions[1], player_actions[2]]
        elif self.n_players == 4:
            return reward_matrix[state, player_actions[0], player_actions[1], player_actions[2], player_actions[3]]
        else:
            Exception("The number of players must be 2, 3 or 4")

    # getting the expected payoff in the future state the arguments must be
    # player1_strategy, player2_strategy, payoff_matrix
    def expectedPayoff(self, payoff_matrix, player_strategies):
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

    # return a copy of the qTable in the state "state"
    def copyQTable(self, qTable: QTable, state: Game, actions):
        qTable = qTable.getQTable()
        if self.env.NPlayers > 4:
            raise Exception("The number of players must be 2, 3 or 4")
        return qTable[self.env.getGameIndex(state)][tuple(actions)].copy()

    # state must be dereferenced earlier than the other dimensions, being the qtable a dictionary qtable[state][...]...
    
    # update the qTable for 2 players
    def updateQTable2(self, qTable: QTable, actions: np.array, alfa: float, gamma: float, r: np.array, next_qVal: np.array):
        state = self.env.getCurrentGame()
        qTable = qTable.getQTable()
        for x in range(self.env.NPlayers):
            qTable[self.env.getGameIndex(state)][actions[0], actions[1], x] = (
                1 - alfa) * qTable[self.env.getGameIndex(state)][actions[0], actions[1], x] + alfa * (r[x] + gamma * next_qVal[x])

    # update the qTable for 3 players
    def updateQTable3(self, qTable: QTable, actions: np.array, alfa: float, gamma: float, r: np.array, next_qVal: np.array):
        state = self.env.getCurrentGame()
        qTable = qTable.getQTable()
        for x in range(self.env.NPlayers):
            qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], x] = (
                1 - alfa) * qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], x] + alfa * (r[x] + gamma * next_qVal[x])

    # update the qTable for 4 players
    def updateQTable4(self, qTable: QTable, actions: np.array, alfa: float, gamma: float, r: np.array, next_qVal: np.array):
        state = self.env.getCurrentGame()
        qTable = qTable.getQTable()
        for x in range(self.env.NPlayers):
            qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], actions[3], x] = (
                1 - alfa) * qTable[self.env.getGameIndex(state)][actions[0], actions[1], actions[2], actions[3], x] + alfa * (r[x] + gamma * next_qVal[x])

    # returns the difference between the two qTables
    def diffQTable(self, newTable: np.array, oldTable: np.array, actions: np.array):
        if self.env.NPlayers > 4:
            raise Exception("The number of players must be 2, 3 or 4")
        return newTable[tuple(actions)] - oldTable

    # attach an observer to the NashQLearning
    def attach(self, observer: NashQLearningObserver):
        self.observers.append(observer)
        if (self.history != None and self.NashQRewards != None and self.NashQRewards != []):
            observer.update(self.history, self.NashQRewards)

    # detach an observer from the NashQLearning
    def detach(self, observer: NashQLearningObserver):
        self.observers.remove(observer)

    # notify the observers
    def notify(self, history: History, NashQRewards):
        for observer in self.observers:
            observer.update(history, NashQRewards)

    # returns the widgets
    def getDisplayable(self):
        return self.widget.getDisplayable()

    # returns the widget
    def getWidget(self):
        return self.widget



