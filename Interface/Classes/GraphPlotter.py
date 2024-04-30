import networkx as nx
from netgraph import Graph
import matplotlib.pyplot as plt

class PlotGraph:
    def __init__(self, graph, ax, edge_labels=None, node_colors=None):
        self.graph = graph
        self.ax = ax
        self.edge_labels = {}
        if edge_labels is not None:
            self.edge_labels = edge_labels
        self.node_colors = {}
        if node_colors is not None:
            self.node_colors = node_colors

    def plot(self):
        Graph(G, node_labels=True, node_layout='circular', edge_labels=self.edge_labels, edge_label_fontdict=dict(size=5, fontweight='bold'), edge_layout='arc', node_size=6, edge_width=0.5, arrows=True, ax=self.ax, node_edge_color=self.node_colors, node_label_fontdict=dict(size=10), edge_label_position=0.2, edge_labels_rotate=False)