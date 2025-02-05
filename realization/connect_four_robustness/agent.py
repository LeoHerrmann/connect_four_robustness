class Agent:
    def __init__(self, name):
        self.name = name

    def determine_action(self, observation: dict) -> int:
        raise NotImplementedError
