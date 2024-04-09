import ipywidgets as widgets
from Classes.EditableMatrix import EditableMatrix
from Classes.PlayerSelection import PlayerSelection
import numpy as np


class TransitionSelection:
    def __init__(self, transitions: np.array = [], playerSelection: PlayerSelection = PlayerSelection(), Q: int = 2):
        self.Q = Q
        self.playerSelection = playerSelection

        self.title = widgets.Label("Transition Selection", layout=widgets.Layout(
            width='100%', justify_content='center')
        )

        self.actionSelections = [PlayerActionSelection(i, actionSpace, lambda x: self.computeSelectedIndex()) for (
            i, actionSpace) in enumerate(playerSelection.getData())]

        self.actionSelectionsContainer = widgets.VBox(
            [actionSelection.getWidget() for actionSelection in self.actionSelections])

        nPlayers = playerSelection.getNPlayers()
        actions = np.prod(playerSelection.getData())
        linearShape = (actions,) + (Q,) * nPlayers
        self.matrices = [EditableMatrix(Q, Q, 1, matrix, editableShape=False)
                         for matrix in transitions.reshape(linearShape)]
        self.matricesStack = widgets.Stack(
            [matrix.get_widget() for matrix in self.matrices], selected_index=0)

        pass

    def computeSelectedIndex(self):
        selectedIndex = 0
        for i, actionSelection in enumerate(self.actionSelections):
            selectedIndex = selectedIndex * actionSelection.actionSpace + \
                actionSelection.getTargetAction()
        self.matricesStack.selected_index = selectedIndex

    def getWidget(self):
        return widgets.VBox([self.title, self.actionSelectionsContainer, self.matricesStack])

    def updatePlayerSelection(self):
        self.actionSelections = [PlayerActionSelection(i, actionSpace, lambda x: self.computeSelectedIndex()) for (
            i, actionSpace) in enumerate(self.playerSelection.getData())]
        self.actionSelectionsContainer.children = [
            actionSelection.getWidget() for actionSelection in self.actionSelections]
        self.computeSelectedIndex()

        # nPlayers = self.playerSelection.getNPlayers()
        # actions = np.prod(self.playerSelection.getData())
        # linearShape = (actions,) + (Q,) * nPlayers
        # self.matricesStack.children = [EditableMatrix(Q, Q, 1, matrix, editableShape=False).get_widget(
        # ) for matrix in transitions.reshape(linearShape)]

    def getData(self):
        targetShape = [
            actionSelection.actionSpace for actionSelection in self.actionSelections] + [self.Q] * self.playerSelection.getNPlayers()
        return np.array([matrix.get_data() for matrix in self.matrices]).reshape(
            targetShape)


class PlayerActionSelection:
    def __init__(self, targetPlayer: int, actionSpace: int, on_change=lambda x: print(x['new'])):
        self.targetPlayer = targetPlayer
        self.actionSpace = actionSpace
        self.on_change = on_change

        self.label = widgets.Label(f"Player {targetPlayer} Action:")
        self.dropdown = widgets.Dropdown(
            options=[i for i in range(actionSpace)])
        self.dropdown.observe(on_change, names='value')
        self.widget = widgets.HBox([self.label, self.dropdown])

    def getTargetAction(self):
        return self.dropdown.value

    def getWidget(self):
        return self.widget
