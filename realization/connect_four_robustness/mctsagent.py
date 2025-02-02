# Code from @flaviendeseure (https://github.com/flaviendeseure/connect4_rl_agent/blob/main/projet/agent/mcts.py)

import numpy as np
import custom_connect_four_v3
#from pettingzoo.classic import connect_four_v3

from agent import Agent


class MctsNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.children = []
        self.action = action
        self.visit_count = 0
        self.total_reward = 0
        self.average_reward = 0


class MctsAgent(Agent):
    def __init__(self, agent_name, is_first_player, n_simulations=100, c_puct=1.0):
        super().__init__()
        self.agent_name = agent_name
        self.is_first_player = is_first_player
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.root_node = MctsNode()

    def get_action(self, observation):
        if sum(observation["action_mask"]) == 0:
            print("Seltsam")

        for _ in range(self.n_simulations):
            leaf_node, leaf_state = self.select(self.root_node, observation["observation"])
            reward = self.rollout(leaf_state)
            self.backpropagate(leaf_node, reward)

        best_child = self.get_best_child(self.root_node)
        return best_child.action

    def reset(self):
        self.root_node = MctsNode()

    def select(self, node: MctsNode, observation: np.ndarray) -> tuple:
        env = custom_connect_four_v3.env()
        env.reset(options={"reverse_order": not self.is_first_player})
        env.state = observation.copy()

        # done = False
        _, _, done, _, _ = env.last()

        while not done:
            if len(node.children) == 0:
                self.expand(node, env)

            node = self.get_best_child(node)

            _, _, done, _, _ = env.last()  # Call env.last() to get the done value

            if not done:
                env.step(node.action)
                _, _, done, _, _ = env.last()  # Call env.last() to get the done value

        return node, env.state

    def rollout(self, state: np.ndarray) -> float:
        env = custom_connect_four_v3.env()
        env.reset(options={"reverse_order": not self.is_first_player})
        env.state = state.copy()

        done = False

        while not done:
            observation, _, _, _, _ = env.last()
            legal_actions = self.get_legal_actions(observation)
            action = np.random.choice(legal_actions)
            env.step(action)
            _, _, done, _, _ = env.last()  # Call env.last() to get the done value

        return self.get_reward(env)

    def backpropagate(self, node: MctsNode, reward: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node.average_reward = node.total_reward / node.visit_count
            node = node.parent

    def expand(self, node: MctsNode, env) -> None:
        observation, _, _, _, _ = env.last()
        legal_actions = self.get_legal_actions(observation)

        if legal_actions == []:
            print("Achtung, ich glaub, gleich gibts nen Fehler")

        for action in legal_actions:
            child_node = MctsNode(parent=node, action=action)
            node.children.append(child_node)

    def get_legal_actions(self, observation) -> list:
        action_mask = observation["action_mask"]
        legal_actions = [i for i, is_legal in enumerate(action_mask) if is_legal]

        if legal_actions == []:
            print("Oh, das sieht gefährlich aus")

        return legal_actions

    def get_best_child(self, node: MctsNode) -> MctsNode:
        return max(node.children, key=lambda child: child.average_reward)

    def get_reward(self, env) -> float:
        _, rewards, done, _, _ = env.last()
        if done:
            return rewards
        return 0
