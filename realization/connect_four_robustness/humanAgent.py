import random
from agent import Agent


class HumanAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def determine_action(self, observation: dict) -> int:
        action_mask = observation["action_mask"]
        legal_actions = [i for i, is_legal in enumerate(action_mask) if is_legal]

        chosen_action = -1

        while chosen_action not in legal_actions:
            print("Folgende Aktionen sind möglich: ", legal_actions)
            chosen_action = int(input("Welche Aktion wählst du? "))

            if chosen_action not in legal_actions:
                print("Die gewählte Aktion ist nicht möglich.")

        return chosen_action
