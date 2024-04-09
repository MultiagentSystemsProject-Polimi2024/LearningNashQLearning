import ipywidgets as widgets
from Classes.PlayerSelection import PlayerSelection
import numpy as np


class PayoffSelection:

    def __init__(self, payoffMatices, playerSelection: PlayerSelection, Q: int = 4) -> None:
        self.payoffMatices = payoffMatices
        self.playerSelection = playerSelection
        self.Q = Q

        self.title = widgets.Label("Payoff Selection", layout=widgets.Layout(
            width='100%', justify_content='center')
        )

        self.valueInputs = [PayoffDisplay(
            i, playerSelection, self) for i in range(self.Q)]

        self.tabs = widgets.Tab(
            [
                widgets.VBox([
                    ActionTupleSelection(
                        i, playerSelection, self.valueInputs[i]).getWidget(),
                    self.valueInputs[i].getWidget()
                ])
                for i in range(self.Q)],
            titles=[f"Game {i}" for i in range(self.Q)]
        )

        pass

    def getWidget(self):
        return widgets.VBox([self.title, self.tabs])


class PayoffDisplay:
    def __init__(self, targetGame: int, playerSelection: PlayerSelection, payoffSelection: PayoffSelection) -> None:
        self.targetGame = targetGame
        self.playerSelection = playerSelection
        self.payoffSelection = payoffSelection
        self.widget = widgets.HBox([
            self.getInput(p)
            for p in range(self.playerSelection.getNPlayers())
        ])

    def getInput(self, value: float = 0):
        return widgets.FloatText(
            value=value,
            description='Payoff:',
            disabled=False
        )

    def updateData(self, data):
        payoff = self.payoffSelection.payoffMatices[self.targetGame][tuple(
            data)]
        for i, value in enumerate(payoff):
            self.widget.children[i].value = value

    def getWidget(self):
        return self.widget


class ActionTupleSelection:
    def __init__(self, targetGame: int, playerSelection: PlayerSelection, targetValueInput: PayoffDisplay) -> None:
        self.targetGame = targetGame
        self.playerSelection = playerSelection
        self.targetValueInput = targetValueInput
        self.widget = widgets.HBox([
            self.getDropdown(p)
            for p in range(self.playerSelection.getNPlayers())
        ])

        self.targetValueInput.updateData(self.getData())

        for dropdown in self.widget.children:
            dropdown.observe(lambda x: self.targetValueInput.updateData(
                self.getData()), names='value')
        pass

    def getDropdown(self, playerIndex: int) -> widgets.Dropdown:
        dropdown = widgets.Dropdown(
            options=range(self.playerSelection.getActionSpace(playerIndex)),
            value=0,
            description=f'Player {playerIndex + 1}:',
            disabled=False,
        )

        return dropdown

    def getWidget(self):
        return self.widget

    def getData(self):
        # print([dropdown.value for dropdown in self.widget.children])
        return [dropdown.value for dropdown in self.widget.children]
