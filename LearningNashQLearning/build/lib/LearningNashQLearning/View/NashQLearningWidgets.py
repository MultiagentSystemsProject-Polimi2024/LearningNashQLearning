import ipywidgets as widgets
import importlib


NashQLearningModule = importlib.import_module(
    '.Model.NashQLearning', package='LearningNashQLearning')

GamesNObserver = NashQLearningModule.GamesNObserver
NashQLearning = NashQLearningModule.NashQLearning


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

    # start the NashQ learning algorithm on button click
    def start(self, b):
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
