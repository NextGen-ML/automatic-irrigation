from random import randint, choice

class DummyAgent:
    def select_action(self, state):
        return [randint(-1, 1)]  # Control only watering