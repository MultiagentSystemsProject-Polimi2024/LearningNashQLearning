import ipywidgets as widgets


class NumberInput():
    def __init__(self, label: str, value, on_change=lambda x: print(x['new'])):
        self.label = widgets.Label(label)
        self.widget = widgets.IntText(value=value)
        self.on_change = on_change
        self.widget.observe(self.on_change, names='value')

    def get_data(self):
        return self.text.value

    def __str__(self) -> str:
        return super().__str__() + f" {self.get_data()}"

    def getWidget(self):
        return widgets.HBox([self.label, self.widget])
