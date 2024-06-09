import ipywidgets as widgets
import numpy as np
from ..Model.Environment import GameObserver, TransitionProfile, EnvironmentObserver
from IPython.display import clear_output, display


class ActionDomainsWidgets(GameObserver):
    def __init__(self, games: list, observeFirst: bool = True, title: str = "Action domains"):
        self.games = games
        self.widgets = []

        self.isUpdating = False

        self.box = widgets.GridBox(layout=widgets.Layout(
            grid_template_columns="repeat(2, 50%)"))

        self.title = widgets.HTML(value=f"<h2>{title}</h2>")

        self.widget = widgets.VBox([self.title, self.box])

        self.update(games[0])

        if (observeFirst):
            games[0].attach(self)

    def update(self, game):
        self.setNPlayers(game.NPlayers, game.possibleActions)

        # check if the possible actions are the same
        if (not np.array_equal(game.possibleActions, self.getPossibleActions())):
            self.setWidgetPossibleActions(game.possibleActions)

    def setNPlayers(self, NPlayers: int, possibleActions: list = None):
        if (NPlayers < len(self.widgets)):
            self.box.children = self.box.children[:NPlayers]
            self.widgets = self.widgets[:NPlayers]
        elif (NPlayers > len(self.widgets)):

            if (possibleActions is not None):
                for i in range(min(NPlayers, len(self.widgets))):
                    self.widgets[i].value = possibleActions[i]

            self.isUpdating = True
            for i in range(max(0, NPlayers - len(self.widgets))):
                newWidget = widgets.IntSlider(
                    value=possibleActions[len(
                        self.widgets)] if possibleActions is not None else 1,
                    min=1,
                    max=10,
                    step=1,
                    description='Player '+str(len(self.widgets))+':',
                )
                self.widgets.append(newWidget)
                newWidget.observe(
                    lambda change: self.setPossibleActions(), names='value')
            self.isUpdating = False

            self.box.children = self.widgets

    def getWidget(self):
        return self.widget

    def getPossibleActions(self):
        return [w.value for w in self.widgets]

    def setWidgetPossibleActions(self, actions):
        self.isUpdating = True
        for i in range(len(actions)):
            self.widgets[i].value = actions[i]
        self.isUpdating = False

    def setPossibleActions(self):
        if (self.isUpdating):
            return

        actions = [w.value for w in self.widgets]

        for game in self.games:
            game.setPossibleActions(actions)

    def setGames(self, games):
        self.games = games


class ActionProfileWidget(GameObserver):
    def __init__(self, game, on_change_callbacks=[]):
        self.game = game
        self.box = widgets.GridBox(layout=widgets.Layout(
            grid_template_columns="repeat(2, 50%)"))
        self.widgets = []

        self.on_change_callbacks = on_change_callbacks

        self.title = widgets.HTML(value="<h2>Action Profile Selection</h2>")

        self.widget = widgets.VBox([self.title, self.box])

        game.attach(self)

        self.update(game)

    def getWidget(self):
        return self.widget

    def get(self):
        return tuple([w.value for w in self.widgets])

    def update(self, game):
        if (game.NPlayers < len(self.widgets)):
            self.box.children = self.box.children[:game.NPlayers]
            self.widgets = self.widgets[:game.NPlayers]
        else:
            for i in range(min(game.NPlayers, len(self.widgets))):
                self.widgets[i].max = game.possibleActions[i]-1

            for i in range(max(0, game.NPlayers - len(self.widgets))):
                newWidget = widgets.IntSlider(
                    value=0,
                    min=0,
                    max=game.possibleActions[len(self.widgets)]-1,
                    step=1,
                    description='Player '+str(len(self.widgets))+':',
                )
                self.widgets.append(newWidget)
                newWidget.observe(self.onChange, names='value')

            self.box.children = self.widgets

    def setOnChanges(self):
        for w in self.widgets:
            w.observe(self.onChange, names='value')

    def addOnChangeCallback(self, callback):
        self.on_change_callbacks.append(callback)
        for w in self.widgets:
            w.observe(callback, names='value')

    def onChange(self, change):
        for callback in self.on_change_callbacks:
            callback(change)


class TransitionProbabilityWidget(GameObserver):
    def __init__(self, game, NGames=2, actionProfile=(0, 0)):
        self.game = game
        self.NGames = NGames
        self.box = widgets.GridBox(layout=widgets.Layout(
            grid_template_columns="repeat(2, 50%)"))
        self.widgets = []
        self.isUpdating = False
        self.actionProfile = actionProfile

        self.normalizeButton = widgets.Button(
            description='Normalize',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Normalize',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )

        self.normalizeButton.on_click(lambda x: self.normalizeProbabilities())

        self.title = widgets.HTML()
        self.updateTitle()

        self.widget = widgets.VBox(
            [self.title, self.box, self.normalizeButton])

        game.attach(self)

        self.setNGames(NGames)

        self.updateActionProfile(actionProfile)

        self.update(game)

    def getWidget(self):
        return self.widget

    def get(self):
        return tuple([w.value for w in self.widgets])

    def update(self, game):
        if (self.game.NPlayers < len(self.actionProfile)):
            self.updateActionProfile(
                tuple(self.actionProfile[:self.game.NPlayers]))
        elif (self.game.NPlayers > len(self.actionProfile)):
            self.updateActionProfile(tuple(np.pad(self.actionProfile, (0, max(
                0, self.game.NPlayers - len(self.actionProfile))), mode='constant', constant_values=0)))

    def normalizeProbabilities(self):
        total = sum([w.value for w in self.widgets])

        if total == 0:
            return

        for w in self.widgets:
            w.value /= total

    def setTransitionProbs(self):
        if (self.isUpdating):
            return

        tp = TransitionProfile({})

        for i, w in enumerate(self.widgets):
            if (w.value > 0):
                tp.setTransition(i, w.value)

        self.game.setTransitionProfile(self.actionProfile, tp)

    def updateTitle(self):
        self.title.value = "<h2>Transition probabilities - " + \
            str(self.actionProfile)+"</h2>"

    def updateTransitionProbabilities(self, tp: dict):
        self.isUpdating = True
        for i, w in enumerate(self.widgets):
            if (i in tp):
                w.value = tp[i]
            else:
                w.value = 0
        self.isUpdating = False

    def updateActionProfile(self, actionProfile):
        self.actionProfile = actionProfile
        transitionProfile = self.game.getTransition(tuple(actionProfile))
        tp = transitionProfile.getTransitionsDict()
        self.updateTransitionProbabilities(tp)

        self.updateTitle()

    def setNGames(self, NGames):
        self.NGames = NGames

        if (self.NGames < len(self.widgets)):
            self.box.children = self.box.children[:NGames]
            self.widgets = self.widgets[:NGames]
        elif (self.NGames > len(self.widgets)):
            for i in range(max(0, self.NGames - len(self.widgets))):
                newWidget = widgets.BoundedFloatText(
                    value=0,
                    min=0,
                    max=1,
                    step=0.01,
                    description='Game '+str(len(self.widgets))+':',
                )
                self.widgets.append(newWidget)
                newWidget.observe(lambda change:
                                  self.setTransitionProbs(),
                                  names='value')

            self.box.children = self.widgets
            self.updateActionProfile(self.actionProfile)


class PayoffWidget(GameObserver):
    def __init__(self, game, actionProfile=(0, 0, 0)):
        self.game = game
        self.box = widgets.GridBox(layout=widgets.Layout(
            grid_template_columns="repeat(2, 50%)"))
        self.widgets = []
        self.actionProfile = actionProfile

        self.isUpdating = False

        self.title = widgets.HTML()
        self.updateTitle()

        self.widget = widgets.VBox([self.title, self.box])

        game.attach(self)

        self.update(game)

    def getWidget(self):
        return self.widget

    def get(self):
        return tuple([w.value for w in self.widgets])

    def update(self, game):
        if (len(self.widgets) > game.NPlayers):
            self.widgets = self.widgets[:game.NPlayers]
            self.box.children = self.box.children[:game.NPlayers]

            self.updateActionProfile(self.actionProfile[:game.NPlayers])

        elif (len(self.widgets) < game.NPlayers):
            for i in range(0, game.NPlayers - len(self.widgets)):
                newWidget = widgets.FloatText(
                    value=0,
                    min=0,
                    max=1,
                    step=0.01,
                    description='Player '+str(len(self.widgets))+':',
                )
                self.widgets.append(newWidget)
                newWidget.observe(lambda change:
                                  self.setPayoff(),
                                  names='value')
            self.box.children = self.widgets

            self.updateActionProfile(tuple(np.pad(self.actionProfile, (0, max(
                0, game.NPlayers - len(self.actionProfile))), mode='constant', constant_values=0)))

            # self.updateActionProfile(self.actionProfile)

    def setPayoff(self):
        if (self.isUpdating):
            return

        payoffs = [w.value for w in self.widgets]

        self.game.setPayoff(self.actionProfile, payoffs)

    def updatePayoffs(self, payoffs):
        self.isUpdating = True

        for i, w in enumerate(self.widgets):
            w.value = payoffs[i]

        self.isUpdating = False

    def updateActionProfile(self, actionProfile):
        self.actionProfile = actionProfile
        self.updateTitle()
        payoff = self.game.getPayoff(self.actionProfile)
        self.updatePayoffs(payoff)

    def updateTitle(self):
        self.title.value = "<h2>Payoffs - "+str(self.actionProfile)+"</h2>"

    def getWidget(self):
        return self.widget


class GameEditor:
    nPlayers = 1

    def __init__(self, game, NGames=2, actionProfile=(0, 0)):
        self.game = game
        self.nPlayers = game.NPlayers

        self.title = widgets.HTML(value="<h1>Game Editor</h1>")

        self.actionDomainsWidgets = ActionDomainsWidgets([game])

        self.actionProfileWidget = ActionProfileWidget(game, [])

        self.TransitionProbabilityWidget = TransitionProbabilityWidget(
            game, NGames, self.actionProfileWidget.get())
        self.PayoffWidget = PayoffWidget(game, self.actionProfileWidget.get())

        self.actionProfileWidget.addOnChangeCallback(
            lambda x: self.TransitionProbabilityWidget.updateActionProfile(self.actionProfileWidget.get()))
        self.actionProfileWidget.addOnChangeCallback(
            lambda x: self.PayoffWidget.updateActionProfile(self.actionProfileWidget.get()))

        self.box = widgets.VBox([
            self.title,
            self.actionDomainsWidgets.getWidget(),
            self.actionProfileWidget.getWidget(),
            self.TransitionProbabilityWidget.getWidget(),
            self.PayoffWidget.getWidget()
        ])

    def getWidget(self):
        return self.box


class EnvironmentWidget(EnvironmentObserver):
    def __init__(self, env) -> None:
        self.env = env

        env.attach(self)

        self.globalActionDomainWidget = ActionDomainsWidgets(
            env.getGames(), False, "Global Action Domain")

        self.NPlayerWidget = widgets.IntSlider(
            value=env.NPlayers,
            min=2,
            max=4,
            step=1,
            description='# Players:',
        )
        self.NPlayerWidget.observe(
            lambda change: env.setNPlayers(change.new), names='value')
        self.NPlayerWidget.observe(
            lambda change: self.globalActionDomainWidget.setNPlayers(change.new), names='value')

        self.NGamesWidget = widgets.IntSlider(
            value=env.getNGames(),
            min=1,
            max=5,
            step=1,
            description='# Games:',
        )

        self.NGamesWidget.observe(lambda change: env.setNGames(
            change.new, self.getGloabalActionDomain()), names='value')

        self.widgets = [
            GameEditor(game, env.getNGames()) for game in env.getGames()
        ]

        self.box = widgets.Tab(
            children=[w.getWidget() for w in self.widgets],
        )

        self.box.titles = ["Game " + str(i) for i in range(len(self.widgets))]

        self.widget = widgets.VBox([
            self.NPlayerWidget,
            self.NGamesWidget,
            self.globalActionDomainWidget.getWidget(),
            self.box
        ])
        self.gameSelector = widgets.Dropdown(
            options=[i for i in range(len(self.widgets))],
            value= 0,
            description='Game:',
        )

        # self.gameSelector = widgets.Select(

        # self.gameSelector = widgets.IntSlider(
        #     value=0,
        #     min=0,
        #     max=len(self.widgets)-1,
        #     step=1,
        #     description='Game:'
        # )
        self.gameSelector.observe(lambda change: self.changeGameShowing(change.new), names='value')
        self.gameOutput = self.widgets[0].getWidget()

        self.secondWidget = widgets.VBox([self.gameSelector, self.gameOutput])
        self.outputWidget = widgets.Output()

    def getWidget(self):
        return self.widget

    def getWidget(self):
        return self.widget

    def getGloabalActionDomain(self):
        return self.globalActionDomainWidget.getPossibleActions()

    def updateEnv(self, env):
        self.NPlayerWidget.value = env.NPlayers
        games = env.getGames()
        for _ in range(max(0, env.getNGames() - len(self.widgets))):
            self.widgets.append(GameEditor(games[len(self.widgets)]))

        self.widgets = self.widgets[:env.getNGames()]

        for w in self.widgets:
            w.TransitionProbabilityWidget.setNGames(env.getNGames())

        self.box.children = [w.getWidget() for w in self.widgets]
        self.box.titles = ["Game " + str(i) for i in range(len(self.widgets))]

        self.gameSelector.options = [i for i in range(len(self.widgets))]

        self.globalActionDomainWidget.setGames(games)

    def changeGameShowing(self, gameIndex):
        # with self.gameOutput:
        #     clear_output()
        #     display(self.widgets[gameIndex].getWidget())
        self.gameOutput = self.widgets[gameIndex].getWidget()
        # clear_output(wait = True)
        # display(self.getGameSelector())
        self.secondWidget.children = [self.gameSelector, self.gameOutput]
        self.displayGameSelector()

    def getWidgetsWithoutGame(self):
        return widgets.VBox([
            self.NPlayerWidget,
            self.NGamesWidget,
            self.globalActionDomainWidget.getWidget()])
        
    def getGameSelector(self):
        return self.secondWidget
    
    def displayGameSelector(self):
        with self.outputWidget:
            clear_output(wait = False)
            display(self.secondWidget)

    def getOutputWidget(self):
        return self.outputWidget
