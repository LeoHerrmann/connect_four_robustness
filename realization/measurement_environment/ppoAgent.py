from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import pettingzoo.utils
from pettingzoo.classic import connect_four_v3
import time
from agent import Agent


class PpoAgent(Agent):
    def __init__(self, name: str, parameters_file_path: str):
        super().__init__(name)
        self.parameters_file_path = parameters_file_path
        self.model = MaskablePPO.load(parameters_file_path)

    def determine_action(self, observation: dict) -> int:
        action_mask = observation["action_mask"]

        chosen_action = int(
            self.model.predict(
                observation["observation"], action_masks=action_mask, deterministic=False
            )[0]
        )
        return chosen_action

