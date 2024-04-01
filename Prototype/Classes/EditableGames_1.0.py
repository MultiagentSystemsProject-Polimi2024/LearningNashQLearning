import ipywidgets as widgets
from Classes.EditableMatrix import EditableMatrix


class EditableGames:
    def __init__(self, n_games) -> None:
        self.n_games = n_games
        self.games = [EditableMatrix(2, 2, output=None)
                      for _ in range(n_games)]
        pass

    def get_widget(self):
        add_button = widgets.Button(description='Add game')
        remove_button = widgets.Button(description='Remove game')
        add_button.on_click(lambda _: self.add_game())
        remove_button.on_click(lambda _: self.remove_game())

        self.tab = widgets.Tab()
        self.tab.children = [game.get_widget() for game in self.games]
        for i in range(self.n_games):
            self.tab.set_title(i, f'Game {i+1}')

        return widgets.VBox([self.tab,
                             widgets.HBox([add_button, remove_button])
                             ])

    def add_game(self):
        self.n_games += 1
        self.games.append(EditableMatrix(2, 2, output=None))
        self.tab.children += (self.games[-1].get_widget(),)
        self.tab.set_title(self.n_games-1, f'Game {self.n_games}')

    def remove_game(self):
        self.n_games -= 1
        self.games.pop()
        self.tab.children = self.tab.children[:-1]
