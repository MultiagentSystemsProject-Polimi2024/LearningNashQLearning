class History:
    def __init__(self):
        self.history = {}
    
    def add(self, key, value):
        self.history[key] = value

    def get(self, key):
        return self.history[key]
    
    def getHistory(self):
        return self.history.copy()
    
    def __str__(self)->str:
        string = ""
        for k in self.history.keys():
            string += str(k) + ": " + str(self.history[k]) + "\n"
            print(k, self.history[k])

        return string