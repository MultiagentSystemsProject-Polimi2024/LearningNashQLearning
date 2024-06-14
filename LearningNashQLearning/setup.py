from setuptools import setup, find_packages

setup(
    name='LearningNashQLearning',
    version='0.12',
    packages=find_packages(),
    description='A library for learning NashQ-learning',
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'networkx',
        'pygambit',
        'netgraph',
        'ipywidgets',
        'notebook',
        'ipympl',
    ]
)
