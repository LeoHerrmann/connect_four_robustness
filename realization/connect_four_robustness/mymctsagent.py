import custom_connect_four_v3
import random
import math
from agent import Agent
# from pettingzoo.classic import connect_four_v3
# from custom_tictactoe import tictactoe


class MyMctsNode:
    id_counter = 0

    def __init__(self, state, selfIsNextPlayer: bool, parent=None, action=None):
        MyMctsNode.id_counter += 1

        self.id: int = self.id_counter
        self.parent: MyMctsNode = parent
        self.children: list[MyMctsNode] = []
        self.state = state
        self.selfIsNextPlayer = selfIsNextPlayer
        self.action: int = action
        self.visit_count: int = 0
        self.total_reward: float = 0

    def pretty_state(self):
        pretty_field = ["_"] * (6 * 7)

        for row_index in range(len(self.state)):
            for field_index in range(len(self.state[row_index])):
                field = self.state[row_index][field_index]
                if field[0] == 1 and field[1] == 0:
                    pretty_field[(row_index * 7) + field_index] = "1"
                elif field[0] == 0 and field[1] == 1:
                    pretty_field[(row_index * 7) + field_index] = "2"

        return pretty_field

    def print_pretty_state(self):
        pretty_field = self.pretty_state()

        for i in range(len(pretty_field)):
            if i % 7 == 0:
                print("\n")

            print(pretty_field[i], end=" ")

#        for column_index in range(len(self.state)):
#            for field_index in range(len(self.state[column_index])):
#                field = self.state[column_index][field_index]
#                if field[0] == 0 and field[1] == 0:
#                    pretty_field[column_index + 2 * column_index + field_index] = "_"
#                elif field[0] == 1 and field[1] == 0:
#                    pretty_field[column_index + 2 * column_index + field_index] = "X"
#                elif field[0] == 0 and field[1] == 1:
#                    pretty_field[column_index + 2 * column_index + field_index] = "O"

#        return pretty_field

class MyMctsAgent(Agent):
    def __init__(self, name, is_first_player, n_simulations=15000, c_uct=1.5):
        super().__init__(name)
        self.is_first_player = is_first_player
        self.n_simulations = n_simulations
        self.c_uct = c_uct

    def toggle_perspective_of_observed_state(self, observed_state):
        for column_index in range(len(observed_state)):
            for field_index in range(len(observed_state[column_index])):
                field = observed_state[column_index][field_index]

                if field[0] == 1 and field[1] == 0:
                    observed_state[column_index][field_index] = [0, 1]
                elif field[0] == 0 and field[1] == 1:
                    observed_state[column_index][field_index] = [1, 0]

        return observed_state

    def determine_action(self, observation) -> int:
        observation_from_agents_perspective = observation["observation"]

        if self.is_first_player:
            observation_from_global_perspective = observation_from_agents_perspective
        else:
            observation_from_global_perspective = self.toggle_perspective_of_observed_state(observation_from_agents_perspective)

        root_node = MyMctsNode(observation_from_global_perspective, True)

        for _ in range(self.n_simulations):
            node_to_simulate = self.select(root_node)
            result = self.simulate(node_to_simulate)
            self.backpropagate(node_to_simulate, result)

        child_with_highest_visitation_count = self.get_child_with_highest_visitation_count(root_node)

        return child_with_highest_visitation_count.action

    def select(self, root_node: MyMctsNode) -> MyMctsNode:
        node = root_node

        environment = custom_connect_four_v3.env()
        # environment = tictactoe.env()

        environment.reset(options={
            # "reverse_order": not self.is_first_player,
            "state": node.state.copy(),
            "nextPlayerIsFirstPlayer": node.selfIsNextPlayer == self.is_first_player
        })

        observation, _, termination, _, _ = environment.last()
        action_mask = observation["action_mask"]
        node_is_fully_expanded = sum(action_mask) == len(node.children)

        while node_is_fully_expanded and not termination:
            node = self.get_child_with_best_uct_score(node)

            environment.reset(options={
                # "reverse_order": not self.is_first_player,
                "state": node.state.copy(),
                "nextPlayerIsFirstPlayer": node.selfIsNextPlayer == self.is_first_player
            })
            observation, _, termination, _, _ = environment.last()
            action_mask = observation["action_mask"]
            node_is_fully_expanded = sum(action_mask) == len(node.children)

        if termination:
            return node
        else:
            already_expanded_actions = [child.action for child in node.children]
            possible_not_taken_actions = []

            for i in range(len(action_mask)):
                if action_mask[i] == 1 and i not in already_expanded_actions:
                    possible_not_taken_actions.append(i)

            randomly_chosen_action = random.choice(possible_not_taken_actions)
            environment.step(randomly_chosen_action)
            observation, _, _, _, _ = environment.last()

            observation_from_agents_perspective = observation["observation"]
            observation_from_global_perspective = observation_from_agents_perspective

            if (self.is_first_player and node.selfIsNextPlayer) or (not self.is_first_player and not node.selfIsNextPlayer):
                observation_from_global_perspective = self.toggle_perspective_of_observed_state(observation_from_agents_perspective)

            new_node = MyMctsNode(
                observation_from_global_perspective,
                not node.selfIsNextPlayer,
                parent=node,
                action=randomly_chosen_action
            )

            node.children.append(new_node)

            return new_node

    def simulate(self, leaf_node: MyMctsNode) -> float:
        environment = custom_connect_four_v3.env()
        # environment = tictactoe.env()
        environment.reset(options={
            # "reverse_order": not self.is_first_player,
            "state": leaf_node.state.copy(),
            "nextPlayerIsFirstPlayer": leaf_node.selfIsNextPlayer == self.is_first_player
        })
        observation, _, termination, _, _ = environment.last()

        while not termination:
            action_mask  = observation["action_mask"]
            legal_actions = [i for i, is_legal in enumerate(action_mask) if is_legal]
            chosen_action = random.choice(legal_actions)
            environment.step(chosen_action)
            observation, _, termination, _, _ = environment.last()

        if self.is_first_player:
            return environment.rewards["player_0"]
        else:
            return environment.rewards["player_1"]

    def backpropagate(self, simulated_node: MyMctsNode, reward: float) -> None:
        node = simulated_node

        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def get_child_with_best_uct_score(self, root_node: MyMctsNode) -> MyMctsNode:
        nodes_with_highest_uct_score = []
        highest_uct_score = self.calculate_uct(root_node.children[0])

        for child_node in root_node.children:
            uct_score = self.calculate_uct(child_node)

            if uct_score == highest_uct_score:
                nodes_with_highest_uct_score.append(child_node)

            if uct_score > highest_uct_score:
                highest_uct_score = uct_score
                nodes_with_highest_uct_score = [child_node]

        return random.choice(nodes_with_highest_uct_score)

    def get_child_with_highest_visitation_count(self, root_node: MyMctsNode) -> MyMctsNode:
        nodes_with_highest_visitation_count = []
        highest_visitation_count = root_node.children[0].visit_count

        for child_node in root_node.children:
            visitation_count = child_node.visit_count

            if visitation_count == highest_visitation_count:
                nodes_with_highest_visitation_count.append(child_node)

            if visitation_count > highest_visitation_count:
                highest_visitation_count = visitation_count
                nodes_with_highest_visitation_count = [child_node]

        return random.choice(nodes_with_highest_visitation_count)

    def calculate_uct(self, node: MyMctsNode) -> float:
        exploration = math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
        return node.total_reward / node.visit_count + self.c_uct * exploration
