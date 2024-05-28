from setuptools import setup, find_packages

setup(
    name='LearningNashQLearning',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'networkx',
        'pygambit',
        'netgraph'
    ]
)
