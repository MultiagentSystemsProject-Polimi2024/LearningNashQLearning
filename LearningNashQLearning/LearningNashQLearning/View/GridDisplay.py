import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class GridDisplay:
    GRID_SIZE = 1

    PLAYER_COLORS = ["red", "blue"]

    def __init__(self, ax: plt.Axes, fig: plt.Figure):
        self.ax = ax
        self.fig = fig
        self.rects = [
            patches.Rectangle(((i - 1) * GridDisplay.GRID_SIZE - GridDisplay.GRID_SIZE/2, -GridDisplay.GRID_SIZE/2), GridDisplay.GRID_SIZE,
                              GridDisplay.GRID_SIZE, linewidth=1, edgecolor="black", fill=False)
            for i in range(3)
        ]

        self.rects += [
            patches.Rectangle((- GridDisplay.GRID_SIZE/2, GridDisplay.GRID_SIZE/2), GridDisplay.GRID_SIZE,
                              GridDisplay.GRID_SIZE, linewidth=1, edgecolor="black", fill=False)
        ]

    def drawGrid(self):
        for i, rect in enumerate(self.rects):
            self.ax.add_patch(rect)
            # draw a text in the center of the rect
            self.ax.text(rect.get_x() + rect.get_width()/2, rect.get_y() + rect.get_height()/2, str(i),
                         fontsize=20, ha='center', va='center', alpha=0.5)
        self.ax.set_xlim(-2 * GridDisplay.GRID_SIZE, 2 * GridDisplay.GRID_SIZE)
        self.ax.set_ylim(-2 * GridDisplay.GRID_SIZE, 2 * GridDisplay.GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

    def gridToCoord(self, gridNum: int):
        return ((gridNum % 3 - 1 + gridNum // 3) * GridDisplay.GRID_SIZE, gridNum // 3 * GridDisplay.GRID_SIZE)

    def drawPlayer(self, gridNum: int, player: int):
        self.ax.add_patch(
            patches.Circle(self.gridToCoord(gridNum), GridDisplay.GRID_SIZE/4, color=GridDisplay.PLAYER_COLORS[player]))

    def drawPlayerAction(self, gridNum: int, player: int, actionProfile: np.ndarray):
        # plot an arrow in the right direction
        dx = 0 if actionProfile[player] != 1 else GridDisplay.GRID_SIZE/2
        if (player == 1):
            dx = -dx

        dy = 0 if actionProfile[player] == 1 else GridDisplay.GRID_SIZE/2
        if (actionProfile[player] == 2):
            dy = -dy

        self.ax.arrow(*self.gridToCoord(gridNum), dx, dy, head_width=0.1, head_length=0.1,
                      fc=GridDisplay.PLAYER_COLORS[player], ec=GridDisplay.PLAYER_COLORS[player])

    def unpackState(self, state: int):
        return (state // 4, state % 4)

    def plotState(self, state: int, actionProfile: np.ndarray):
        self.ax.clear()
        self.drawGrid()
        for player in range(2):
            self.drawPlayer(self.unpackState(state)[player], player)
            self.drawPlayerAction(self.unpackState(
                state)[player], player, actionProfile)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
