import random
from agent import Agent


class RandomAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def determine_action(self, observation: dict) -> int:
        action_mask = observation["action_mask"]
        legal_actions = [i for i, is_legal in enumerate(action_mask) if is_legal]
        chosen_action = random.choice(legal_actions)
        return chosen_action
