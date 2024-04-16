import numpy as np
import random as rand
import nashpy as nash
import ipywidgets as widgets
import IPython.display as display

class NahQLearning:
    def __init__(self, n_players, n_games, action_per_player, transition_matrix, reward_matrix) -> None:
        self.n_players = n_players
        self.n_games = n_games
        self.action_per_player = action_per_player
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix 
        
        self.gamesLoadingBarNashQ = widgets.IntProgress(
        value=0,
        min=0,
        max=n_games-1,
        step=1,
        description='Games:',
        bar_style='info',
        ) 
        display(self.gamesLoadingBarNashQ)

    def computeNashEq(self, state, payoff_matrixA, payoff_matrixB):
        game = nash.Game(payoff_matrixA[state, :, :, 0], payoff_matrixB[state, :, :, 1])
        eqs = game.vertex_enumeration()
        try:
            eq = next(eqs)
            return np.abs(eq)
        except Exception:
            return [[1, 0], [0, 1]]

    def reward(state, player1_action, player2_action, payoffMatrix):
        return payoffMatrix[state, player1_action, player2_action]