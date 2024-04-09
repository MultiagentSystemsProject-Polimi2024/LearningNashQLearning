import ipywidgets as widgets


class PlayerSelection():
    def __init__(self, playersStartingActions: list = [], defultActionSpace: int = 2):
        self.defultActionSpace = defultActionSpace

        self.title = widgets.Label("Player Selection", layout=widgets.Layout(
            width='100%', justify_content='center')
        )

        self.addButton = widgets.Button(description="", icon='plus', layout=widgets.Layout(
            width='30px', border_radius='50%')
        )
        self.addButton.on_click(lambda x: self.addPlayer())

        self.playerWidgets = []
        self.playerWidgetsContainer = widgets.Box([self.addButton], layout=widgets.Layout(
            width='100%', flex_flow='column', justify_content='center', align_items='center', display='flex'))

        for actionSpace in playersStartingActions:
            self.addPlayer(actionSpace)

        pass

    def addPlayer(self, actionSpace: int = -1):
        if actionSpace == -1:
            actionSpace = self.defultActionSpace

        self.playerWidgets.append(PlayerWidget(
            len(self.playerWidgets), actionSpace, self.removePlayer))
        self.playerWidgetsContainer.children = [
            *self.playerWidgetsContainer.children[:-1], self.playerWidgets[-1].getWidget(), self.addButton]

    def removePlayer(self, player):
        self.playerWidgets.remove(player)
        index = self.playerWidgetsContainer.children.index(player.getWidget())
        self.playerWidgetsContainer.children = [
            *self.playerWidgetsContainer.children[:index], *self.playerWidgetsContainer.children[index + 1:]]

    def setActionSpace(self, index: int, actionSpace: int):
        self.playerWidgets[index].setActionSpace(actionSpace)

    def getActionSpace(self, index: int):
        return self.playerWidgets[index].getActionSpace()

    def getData(self):
        return [player.getActionSpace() for player in self.playerWidgets]

    def getNPlayers(self):
        return len(self.playerWidgets)

    def __str__(self) -> str:
        return super().__str__() + f"{self.get_data()}"

    def setupWidgets(self):

        pass

    def getWidget(self):
        return widgets.VBox([
            self.title,
            self.playerWidgetsContainer

        ], layout=widgets.Layout(
            justify_content='center', overflow='hidden')
        )
        pass


class PlayerWidget():
    def __init__(self, index: int, actionSpace: int, removePlayer=lambda x: print(x)):
        self.index = index
        self.actionSpace = actionSpace
        self.removeButton = widgets.Button(description="", icon='minus', layout=widgets.Layout(
            width='50px')
        )
        self.removeButton.on_click(lambda x: removePlayer(self))
        self.setupWidgets()

    def setupWidgets(self):
        self.title = widgets.Label(f"Player {self.index}", layout=widgets.Layout(
            width='100%', justify_content='center'))
        self.label = widgets.Label("Action Space")
        self.actionSpaceWidget = widgets.IntText(value=self.actionSpace, layout=widgets.Layout(
            width='80px', justify_content='center', text_align='center'
        ))
        self.widget = widgets.VBox([self.title, self.label, self.actionSpaceWidget, self.removeButton], layout=widgets.Layout(
            width='100%', justify_content='center', align_items='center', overflow='hidden', min_width='200px', border='1px solid black')
        )

    def setActionSpace(self, actionSpace: int):
        self.actionSpace = actionSpace
        self.actionSpaceWidget.value = actionSpace

    def getActionSpace(self):
        return self.actionSpaceWidget.value

    def getWidget(self):
        return self.widget
