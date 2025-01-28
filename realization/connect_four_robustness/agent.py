# Code from @flaviendeseure (https://github.com/flaviendeseure/connect4_rl_agent/blob/main/projet/agent/base_agent.py)

from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        self.reset()

    @abstractmethod
    def get_action(self, observation: dict) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
