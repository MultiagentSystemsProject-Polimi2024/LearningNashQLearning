import numpy as np
import random as rand
import pygambit as pg
import ipywidgets as widgets
import IPython.display as display

class NahQLearning:
    def __init__(self, n_players, n_games, action_per_player, transition_matrix, reward_matrix, goal_state = none) -> None:
        self.n_players = n_players
        self.n_games = n_games
        self.action_per_player = action_per_player
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix 
        self.goal_state = goal_state 
        
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
        #convert the Nash Equilibrium to an array
        e = np.zeros((self.n_players,self.action_per_player))
        for i in range(self.n_players):
            for j in range(self.action_per_player):
                e[i][j] = (float(eq[str(i+1)][str(j+1)]))
        return e
    
    #computing Nash equilibrium for 3 players
    def computeNashEq(self, state, payoff_matrix):
        if(self.n_players == 3):
            #create the game
            game = pg.Game.from_arrays(payoff_matrix[state,:,:,0], payoff_matrix[state,:,:,1], payoff_matrix[state,:,:,2], title=("gambe number"+str(state)))
        elif(self.n_players == 4):
            game = pg.Game.from_arrays(payoff_matrix[state,:,:,0], payoff_matrix[state,:,:,1], payoff_matrix[state,:,:,2], payoff_matrix[state,:,:,3], title=("gambe number"+str(state)))
        #compute the Nash Equilibrium
        eq = pg.nash.logit_solve(game).equilibria
        #normalize the equilibrium
        eq = eq[0].normalize()
        #convert the Nash Equilibrium to an array
        e = np.zeros((self.n_players,self.action_per_player))
        for i in range(self.n_players):
            for j in range(self.action_per_player):
                e[i][j] = (float(eq[str(i+1)][str(j+1)]))
        return e


    def getNextState(self, state, player_actions):
        if(self.n_players == 2):
            return np.random.choice(range(self.n_games), p=self.transition_matrix[player_actions[0], player_actions[1], state])
        elif(self.n_players == 3):
            return np.random.choice(range(self.n_games), p=self.transition_matrix[player_actions[0], player_actions[1], player_actions[2], state])
        elif(self.n_players == 4):
            return np.random.choice(range(self.n_games), p=self.transition_matrix[player_actions[0], player_actions[1], player_actions[2],player_actions[3], state])
        else:
            Exception("The number of players must be 2, 3 or 4")

    #getting reward for a given state and actions, the arguments must be 
    #reward_matrix, state, player1_action, player2_action, player3_action(optional)
    def reward(self, state, player_actions, reward_matrix):
        if self.n_players == 2:
            return reward_matrix[state, player_actions[0], player_actions[1]]
        elif self.n_players == 3:
            return reward_matrix[state, player_actions[0], player_actions[1], player_actions[2]]
        elif self.n_players == 4:
            return reward_matrix[state, player_actions[0], player_actions[1], player_actions[2], player_actions[3]]
        else:
            Exception("The number of players must be 2, 3 or 4")
    
    #getting the expected payoff in the future state the arguments must be
    #player1_strategy, player2_strategy, payoff_matrix
    def expectedPayoff(self, payoff_matrix, player_strategies):
        if self.n_players == 2:
            return np.dot(player_strategies[0], np.dot(payoff_matrix, player_strategies[1]))
        elif self.n_players == 3:
            return np.dot(player_strategies[0], np.dot(player_strategies[1], np.dot(payoff_matrix, player_strategies[2])))
        elif self.n_players == 4:
            return np.dot(player_strategies[0], np.dot(player_strategies[1], np.dot(player_strategies[2], np.dot(payoff_matrix, player_strategies[3]))))
        else:
            Exception("The number of players must be 2, 3 or 4")
        
    #getting the expected payoff in the future state for 2 players
    def expectedPayoff(self, payoff_matrix, player1_strategy, player2_strategy):
        expected_payoff = np.dot(np.dot(player1_strategy, payoff_matrix), player2_strategy)
        return expected_payoff
    
    #getting the expected payoff in the future state for 3 players
    def expectedPayoff(self, payoff_matrix, player1_strategy, player2_strategy, player3_strategy):
        expected_payoff = np.dot(np.dot(np.dot(player1_strategy, payoff_matrix), player2_strategy), player3_strategy)
        return expected_payoff


    #NashQ learning algorithm for 2 players
    def nashQlearning(self, episodes, alfa, gamma, epsilon, pure_training_ep, decaying_epsilon, reset = False):
        
        #declare qTables
        qTables = [np.zeros((self.n_games, self.action_per_player, self.action_per_player, self.n_players)) for _ in range(self.n_players)]

        #start from first state
        state = 0
        
        #initialize values to display
        totalReward = [np.array([0, 0], dtype=float) for _ in range(self.n_players)]
        diffs = [[]for _ in range(self.n_players)]
        NashQRewards = [[]for _ in range(self.n_players)]
        history = {}

        nashEq = np.zeros((self.n_players, self.action_per_player))


        for t in range(episodes):
            history_element ={}        

            alfa = alfa / (t + 1 - pure_training_ep) if t >= pure_training_ep else alfa
            epsilon = epsilon / (t + 1 - decaying_epsilon) if t >= decaying_epsilon else epsilon

            #choose action
            player1_action = np.random.choice(self.action_per_player, p=nashEq[0]) if np.random.rand() > epsilon else np.random.choice(self.action_per_player)
            player2_action = np.random.choice(self.action_per_player, p=nashEq[1]) if np.random.rand() > epsilon else np.random.choice(self.action_per_player)
            
            #calculating next state
            next_state = np.random.choice(range(self.n_games), p=self.transition_matrix[player1_action, player2_action, state])

            #getting reward for the current move
            r = self.reward(state, player1_action, player2_action, self.reward_matrix)
            
            for i in range(self.n_players):
                #get qTable for the player i
                qTable = qTables[i]


                #compute the expected payoff for the next state
                next_NashEq = self.computeNashEq(next_state, qTable, qTable)
                next_qVal_0 = self.expectedPayoff(qTable[next_state, :, :, 0], next_NashEq[0], next_NashEq[1])
                next_qVal_1 = self.expectedPayoff(qTable[next_state, :, :, 1], next_NashEq[0], next_NashEq[1])
                
                #copy qTable
                oldQ = qTable[state, player1_action, player2_action].copy()
                
                #update qTable
                qTable[state, player1_action, player2_action, 0] = (1 - alfa) * qTable[state, player1_action, player2_action, 0] + alfa * (r[0] + gamma * next_qVal_0)
                qTable[state, player1_action, player2_action, 1] = (1 - alfa) * qTable[state, player1_action, player2_action, 1] + alfa * (r[1] + gamma * next_qVal_1)

                #memorize the qTable
                history_element[('Q'+str(i))] = qTable
                #memorize the difference between the old and the new value in the qTable
                diffs[i].append(qTable[state, player1_action, player2_action] - oldQ)
                
                #update the total reward of the player i
                totalReward[i] += r
                #memorize the reward of the player i
                NashQRewards[i].append(r)
            
            #memorize the state
            history_element['current_state'] = state
            #add the history element to the history
            history[t] = history_element

            #update the state
            if(reset and self.goal_state != None and state == self.goal_state):
                next_state = 0
            else:
                state = next_state

            #update the loading bar
            self.gamesLoadingBarNashQ.value += 1
        return totalReward, diffs, NashQRewards, history
    
    #NashQ learning algorithm for n players
    def nashQlearning(self, episodes, alfa, gamma, epsilon, pure_training_ep, decaying_epsilon, reset = False):
        
        #declare qTables
        qTables = [np.zeros((self.n_games, self.action_per_player, self.action_per_player, self.n_players)) for _ in range(self.n_players)]

        #start from first state
        state = 0
        
        #initialize values to display
        totalReward = [np.array([0, 0], dtype=float) for _ in range(self.n_players)]
        diffs = [[]for _ in range(self.n_players)]
        NashQRewards = [[]for _ in range(self.n_players)]
        history = {}
        nashEq = np.zeros((self.n_players, self.action_per_player))

        for t in range(episodes):
            history_element ={}        

            alfa = alfa / (t + 1 - pure_training_ep) if t >= pure_training_ep else alfa
            epsilon = epsilon / (t + 1 - decaying_epsilon) if t >= decaying_epsilon else epsilon

            player_action = [[]]*self.n_players
            #choose action
            for x in range(self.n_players):
                player_action[x] = np.random.choice(self.action_per_player, p=nashEq[x]) if np.random.rand() > epsilon else np.random.choice(self.action_per_player)
            
            #calculating next state
            next_state = self.getNextState(state, player_action)  

            #getting reward for the current move
            r = self.reward(state, player_action, self.reward_matrix)
            
            for i in range(self.n_players):
                #get qTable for the player i
                qTable = qTables[i]

                #compute the expected payoff for the next state
                next_NashEq = self.computeNashEq(next_state, qTable)
                next_qVal = np.zeros(self.n_players)
                for x in range(self.n_players):
                    next_qVal[x] = self.expectedPayoff(qTable[next_state, :, :, x], next_NashEq)
                
                #copy qTable
                """attention to the number of players!!!!"""
                """define a function to memorize this value"""
                oldQ = qTable[state, player_action[0], player_action[1]].copy()
                
                #update qTable
                for x in range(self.n_players):
                    """define a function to update the qTable"""
                    qTable[state, player1_action, player2_action, 0] = (1 - alfa) * qTable[state, player1_action, player2_action, 0] + alfa * (r[0] + gamma * next_qVal_0)
                    qTable[state, player1_action, player2_action, 1] = (1 - alfa) * qTable[state, player1_action, player2_action, 1] + alfa * (r[1] + gamma * next_qVal_1)

                #memorize the qTable
                history_element[('Q'+str(i))] = qTable

                #memorize the difference between the old and the new value in the qTable
                """attention to the number of players"""
                diffs[i].append(qTable[state, player1_action, player2_action] - oldQ)
                
                #update the total reward of the player i
                totalReward[i] += r
                #memorize the reward of the player i
                NashQRewards[i].append(r)
            

            #memorize the state
            history_element['current_state'] = state
            #add the history element to the history
            history[t] = history_element

            #update the state
            if(reset and self.goal_state != None and state == self.goal_state):
                next_state = 0
            else:
                state = next_state

            #update the loading bar
            self.gamesLoadingBarNashQ.value += 1
        return totalReward, diffs, NashQRewards, history