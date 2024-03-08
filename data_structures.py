"""
There follows a set of data structures for MDP modeling where every state of
the Markov's chain is composed by a Nash game
Freely inspired by mdptoolbox library
https://pymdptoolbox.readthedocs.io/en/latest/api/mdptoolbox.html

note that in all cases actions are strings
"""

"""COMPLEX IMPLEMENTATION"""

"""
MC(states, transitions) -> Markov Chain
A Markov Chain is composed by a set of states and a set of transitions
    states is a list of State objects
    transitions is a dictionary where the keys are the states and the values are dictionaries
    where the keys are the actions and the values are the couples (next_state, probability)
"""
class MC:   
    def __init__(self, states, transitions):
        self.states = states
        self.transitions = transitions

    def possible_actions(self, state):
        return self.transitions[state].keys()


"""
State(name, reward) -> State
every state of the Markov's chain is composed by a Nash game
    name is the name of the state
    reward is a dictionary where the keys are the tuples expressing the combination of actions 
    of the game and the values are the tuples expressing the reward for each player

"""
class State:

    def __init__(self, name, reward):
        self.name = name
        self.reward = reward
    
    def nashEquilibrium():
        "to be completed"
        pass



""""EASY IMPLEMENTATION"""

import matplotlib as plt

"""
MC(states, transitions) -> Markov Chain
A Markov Chain is composed by a set of states and a set of transitions
    states is a list of State objects
    mdp_actions is a list of the possible actions of the MDP, this allows to map every action to its index in
    the transition matrix 
    transitions is a array that has the shape (n_states, n_actions, n_states),
    in every cell lies the probability of transitioning from the state i to the state j
    given the action a
"""
class MC:   
    def __init__(self, states, mdp_actions, transitions):
        self.states = states
        self.mdp_actions = mdp_actions
        self.transitions = transitions

    def possible_actions(self, starting_state):
        """errore"""
        indexes = self.transitions[starting_state, :, :].nonzero() 
        return self.mdp_actions[indexes]


"""
State(name, game_actions, reward) -> State
every state of the Markov's chain is composed by a Nash game
    name is the name of the state
    game_actions is a list of the possible actions of the game (not in the MDP) this allows to map 
    the actions of the game to the corresponding index in the reward matrix
    reward is an array in the shape (n_actions, n_actions) where the cell (i, j) is the
    tuple of the rewards for the players
"""
class State:

    def __init__(self, name, game_actions, reward):
        self.name = name
        self.game_actions = game_actions
        self.reward = reward
        self.reward = reward
    
        def nashEquilibrium():
        "to be completed"
        pass

