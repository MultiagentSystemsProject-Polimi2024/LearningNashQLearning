# This class is used to log the situation of the game at each step, so that it can be used to analyze the game later on
# For each training episode, it saves the current game, the action taken by the agents and the rewards received, the current Q-Tables of the agents and the current policy that the agents use to select their actions
class History:
    def __init__(self):
        self.history = {}
        self.observers = []

    def add(self, key, value):
        self.history[key] = value

        self.__notify()

    def get(self, key):
        return self.history[key]

    def getHistory(self):
        return self.history.copy()

    def __str__(self) -> str:
        string = ""
        for k in self.history.keys():
            string += str(k) + ": " + str(self.history[k]) + "\n"
            print(k, self.history[k])

        return string

    def keys(self):
        return self.history.keys()

    # Methods for Observer Pattern

    def attach(self, observer):
        self.observers.append(observer)

    def detach(self, observer):
        self.observers.remove(observer)

    def __notify(self):
        for observer in self.observers:
            observer.update(self)
