import numpy as np
import random as rand
import pygambit as pg
import ipywidgets as widgets
import IPython.display as display

class NahQLearning:
    def __init__(self, n_players, n_games, action_per_player, transition_matrix, reward_matrix) -> None:
        self.n_players = n_players
        self.n_games = n_games
        self.action_per_player = action_per_player
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix 
        
        #widget
        self.gamesLoadingBarNashQ = widgets.IntProgress(
        value=0,
        min=0,
        max=n_games-1,
        step=1,
        description='Games:',
        bar_style='info',
        ) 
        display(self.gamesLoadingBarNashQ)

    #computing Nash equilibrium for 2 players
    def computeNashEq(self, state, payoff_matrixA, payoff_matrixB):
        #create the game
        game = pg.Game.from_arrays(payoff_matrixA[state,:,:,0], payoff_matrixB[state,:,:,1], title=("gambe number"+str(state)))
        #compute the Nash Equilibrium
        eq = pg.nash.enummixed_solve(game).equilibria
        #normalize the equilibrium
        eq = eq[0].normalize()
        #convert the Nash Equilibrium to a list
        e = [[],[]]
        for i in range(2):
            for j in range(2):
                e[i].append(float(eq[str(i+1)][str(j+1)]))
        return e
    
    #computing Nash equilibrium for 3 players
    def computeNashEq(self, state, payoff_matrixA, payoff_matrixB, payoff_matrixC):
        #create the game
        game = pg.Game.from_arrays(payoff_matrixA[state,:,:,0], payoff_matrixB[state,:,:,1], payoff_matrixC[state,:,:,2], title=("gambe number"+str(state)))
        #compute the Nash Equilibrium
        eq = pg.nash.logit_solve(game).equilibria
        #normalize the equilibrium
        eq = eq[0].normalize()
        #convert the Nash Equilibrium to a list
        e = [[],[],[]]
        for i in range(3):
            for j in range(2):
                e[i].append(float(eq[str(i+1)][str(j+1)]))
        return e


    #getting reward for a given state and actions, the arguments must be 
    #reward_matrix, state, player1_action, player2_action, player3_action(optional)
    def reward(self, **kwargs):
        if self.n_players == 2:
            return kwargs['reward_matrix'][kwargs['state'], kwargs['player1_action'], kwargs['player2_action']]
        elif self.n_players == 3:
            return kwargs['reward_matrix'][kwargs['state'], kwargs['player1_action'], kwargs['player2_action'], kwargs['player3_action']]
        else:
            return None
    
    #getting the expected payoff in the future state the arguments must be
    #player1_strategy, player2_strategy, payoff_matrix
    def expectedPayoff(self, **kwargs):
        if self.n_players == 2:
            return np.dot(kwargs['player1_strategy'], np.dot(kwargs['payoff_matrix'], kwargs['player2_strategy']))
        elif self.n_players == 3:
            return np.dot(kwargs['player1_strategy'], np.dot(kwargs['player2_strategy'], np.dot(kwargs['payoff_matrix'], kwargs['player3_strategy'])))
        else:
            return None
        
    #getting the expected payoff in the future state for 2 players
    def expectedPayoff(self, payoff_matrix, player1_strategy, player2_strategy):
        expected_payoff = np.dot(np.dot(player1_strategy, payoff_matrix), player2_strategy)
        return expected_payoff
    
    #getting the expected payoff in the future state for 3 players
    def expectedPayoff(self, payoff_matrix, player1_strategy, player2_strategy, player3_strategy):
        expected_payoff = np.dot(np.dot(np.dot(player1_strategy, payoff_matrix), player2_strategy), player3_strategy)
        return expected_payoff


    #NashQ learning algorithm
    def nashQlearning(self, episodes, alfa, gamma, epsilon, pure_training_ep, decaying_epsilon):
        
        #declare qTables
        qTables = [np.zeros((self.n_games, self.action_per_player, self.action_per_player, self.n_players)) for _ in range(self.n_players)]

        #start from first state
        state = 0
        
        #initialize values to display
        totalReward = [np.array([0, 0], dtype=float) for _ in range(self.n_players)]
        diffs = [[]for _ in range(self.n_players)]
        NashQRewards = [[]for _ in range(self.n_players)]
        NashEquilibria = [[]for _ in range(self.n_players)]
        nashEq = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]


        for t in range(episodes):
            alfa = alfa / (t + 1 - pure_training_ep) if t >= pure_training_ep else alfa
            epsilon = epsilon / (t + 1 - decaying_epsilon) if t >= decaying_epsilon else epsilon

            player1_action = np.random.choice(self.action_per_player, p=nashEq[0]) if np.random.rand() > epsilon else np.random.choice(self.action_per_player)
            player2_action = np.random.choice(self.action_per_player, p=nashEq[1]) if np.random.rand() > epsilon else np.random.choice(self.action_per_player)

            next_state = np.random.choice(range(self.n_games), p=self.transition_matrix[player1_action, player2_action, state])

            r = self.reward(state, player1_action, player2_action, self.reward_matrix)
            
            for i in range(self.n_players):
                qTable = qTables[i]

                nashEq = np.abs(self.computeNashEq(state, qTable, qTable))

                NashEquilibria[i].append(nashEq)

                next_NashEq = self.computeNashEq(next_state, qTable, qTable)
                next_qVal_0 = self.expectedPayoff(qTable[next_state, :, :, 0], next_NashEq[0], next_NashEq[1])
                next_qVal_1 = self.expectedPayoff(qTable[next_state, :, :, 1], next_NashEq[0], next_NashEq[1])
                
                oldQ = qTable[state, player1_action, player2_action].copy()
                
                qTable[state, player1_action, player2_action, 0] = (1 - alfa) * qTable[state, player1_action, player2_action, 0] + alfa * (r[0] + gamma * next_qVal_0)
                qTable[state, player1_action, player2_action, 1] = (1 - alfa) * qTable[state, player1_action, player2_action, 1] + alfa * (r[1] + gamma * next_qVal_1)

                diffs[i].append(qTable[state, player1_action, player2_action] - oldQ)
                
                totalReward[i] += r
                NashQRewards[i].append(r)
                print(qTables[i])
            state = next_state
            self.gamesLoadingBarNashQ.value += 1
        return totalReward, diffs, NashQRewards, NashEquilibria
