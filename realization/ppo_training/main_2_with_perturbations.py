"""
Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
"""
import time

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
from pettingzoo.classic import connect_four_v3
import random
import numpy as np
import json

from stable_baselines3.common.callbacks import BaseCallback


class ModelWrapper:
    def __init__(self, name: str, model: MaskablePPO):
        self.name = name
        self.model = model
        self.lowest_win_rate = 1.0
        self.strongest_opponent_name = None


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def __init__(self, env, stationary_model: MaskablePPO | None, training_agent_is_player_0: bool):
        super().__init__(env)
        self.stationary_model = stationary_model
        self.training_agent_is_player_0 = training_agent_is_player_0

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        # If training agent is second player, execute first step based on stationary policy
        if not self.training_agent_is_player_0:
            stationary_agent_observation = self.observe(self.agent_selection)
            stationary_agent_action_mask = self.action_mask()

            if self.stationary_model is None:
                # If no stationary model is given execute step based on random policy
                stationary_agent_legal_actions = [i for i, is_legal in enumerate(stationary_agent_action_mask) if is_legal]
                stationary_agent_chosen_action = random.choice(stationary_agent_legal_actions)
            else:
                stationary_agent_chosen_action = int(
                    self.stationary_model.predict(
                        stationary_agent_observation, action_masks=stationary_agent_action_mask, deterministic=False
                    )[0]
                )

            super().step(stationary_agent_chosen_action)

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info.

        The observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).
        """
        training_agent = self.agent_selection

        # Execute step for training player
        super().step(action)

        stationary_agent = self.agent_selection

        # If game is over, return
        if self.terminations[training_agent]:
            return (
                self.observe(stationary_agent),
                self._cumulative_rewards[training_agent],
                self.terminations[training_agent],
                self.truncations[training_agent],
                self.infos[training_agent],
            )

        # Execute step for frozen player
        stationary_agent_observation = self.observe(self.agent_selection)
        stationary_agent_action_mask = self.action_mask()

        # If stationary models are given execute step based on given model
        if self.stationary_model is None:
            # If no stationary models are given execute step based on random policy
            stationary_agent_legal_actions = [i for i, is_legal in enumerate(stationary_agent_action_mask) if is_legal]
            stationary_agent_chosen_action = random.choice(stationary_agent_legal_actions)
        else:
            stationary_agent_chosen_action = int(
                self.stationary_model.predict(
                    stationary_agent_observation, action_masks=stationary_agent_action_mask, deterministic=False
                )[0]
            )

        super().step(stationary_agent_chosen_action)

        return (
            self.observe(training_agent),
            self._cumulative_rewards[training_agent],
            self.terminations[training_agent],
            self.truncations[training_agent],
            self.infos[training_agent],
        )

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


class ValueLossCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.value_losses = []
        self.policy_gradient_losses = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.value_losses.append(self.logger.name_to_value["train/value_loss"])
        self.policy_gradient_losses.append(self.logger.name_to_value["train/policy_gradient_loss"])


def train_action_mask(env_fn, model_tuples: list[tuple[ModelWrapper, ModelWrapper]], steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""

    # Reset lowest win rates and strongest opponents of model wrappers
    for model_wrapper_tuple in model_tuples:
        for model_wrapper in model_wrapper_tuple:
            model_wrapper.lowest_win_rate = 1.0
            model_wrapper.strongest_opponent_name = None

    # Evaluate all models against each other and save the strongest opponent
    for player_0_index in range(len(model_tuples)):
        for player_1_index in range(len(model_tuples)):
            player_0_win_rate, average_game_length = evaluate_model_wrappers_against_each_other(
                env_fn,(model_tuples[player_0_index][0], model_tuples[player_1_index][1])
            )

            player_1_win_rate = 1 - player_0_win_rate

            if player_0_win_rate <= model_tuples[player_0_index][0].lowest_win_rate:
                model_tuples[player_0_index][0].lowest_win_rate = player_0_win_rate
                model_tuples[player_0_index][0].strongest_opponent_name = model_tuples[player_1_index][1].name

            if player_1_win_rate <= model_tuples[player_1_index][1].lowest_win_rate:
                model_tuples[player_1_index][1].lowest_win_rate = player_1_win_rate
                model_tuples[player_1_index][1].strongest_opponent_name = model_tuples[player_0_index][0].name

    deep_copied_model_tuples = save_and_load_model_tuples(model_tuples)

    # Train each model against its strongest opponent
    model_0_value_losses = []
    model_0_policy_gradient_losses = []
    model_1_value_losses = []
    model_1_policy_gradient_losses = []

    for model_tuple in model_tuples:
        # Find the strongest opponent for each model
        stationary_model_for_model_0 = None
        stationary_model_for_model_1 = None

        for deep_copied_model_tuple in deep_copied_model_tuples:
            if deep_copied_model_tuple[1].name == model_tuple[0].strongest_opponent_name:
                stationary_model_for_model_0 = deep_copied_model_tuple[1]
            if deep_copied_model_tuple[0].name == model_tuple[1].strongest_opponent_name:
                stationary_model_for_model_1 = deep_copied_model_tuple[0]

        if stationary_model_for_model_0 is None:
            print("That's weird. Strongest opponent for model 0 was not found...")

        if stationary_model_for_model_1 is None:
            print("That's weird. Strongest opponent for model 1 was not found...")

        # Train each model against its strongest opponent
        env_0 = env_fn.env(**env_kwargs)
        print(f"Starting training of {model_tuple[0].name} against stationary {stationary_model_for_model_0.name} on {str(env_0.metadata['name'])}.")
        env_0 = SB3ActionMaskWrapper(env_0, stationary_model_for_model_0.model, True)
        env_0.reset(seed=seed)  # Must call reset() in order to re-define the spaces
        env_0 = ActionMasker(env_0, mask_fn)  # Wrap to enable masking (SB3 function)
        model_tuple[0].model.set_env(env_0)
        value_loss_0_callback = ValueLossCallback()
        model_tuple[0].model.learn(total_timesteps=steps, callback=value_loss_0_callback, reset_num_timesteps=False)
        # model_tuple[0].model.save(f"weights_from_main_with_perturbations/{model_tuple[0].name}_{time.strftime('%Y%m%d-%H%M%S')}")
        # print(f"Model of {model_tuple[0].name} has been saved.")
        print(f"Finished training on {str(env_0.unwrapped.metadata['name'])}.\n")

        env_0.close()
        model_0_value_losses += value_loss_0_callback.value_losses[1:]
        model_0_policy_gradient_losses += value_loss_0_callback.policy_gradient_losses[1:]

        env_1 = env_fn.env(**env_kwargs)
        print(f"Starting training of {model_tuple[1].name} against stationary {stationary_model_for_model_1.name} on {str(env_0.metadata['name'])}.")
        env_1 = SB3ActionMaskWrapper(env_1, stationary_model_for_model_1.model, False)
        env_1.reset(seed=seed)  # Must call reset() in order to re-define the spaces
        env_1 = ActionMasker(env_1, mask_fn)  # Wrap to enable masking (SB3 function)
        model_tuple[1].model.set_env(env_1)
        value_loss_1_callback = ValueLossCallback()
        model_tuple[1].model.learn(total_timesteps=steps, callback=value_loss_1_callback, reset_num_timesteps=False)
        # model_tuple[1].model.save(f"weights_from_main_with_perturbations/{model_tuple[1].name}_{time.strftime('%Y%m%d-%H%M%S')}")
        # print(f"Model of {model_tuple[1].name} has been saved.")
        print(f"Finished training on {str(env_1.unwrapped.metadata['name'])}.\n")

        env_1.close()
        model_1_value_losses += value_loss_1_callback.value_losses[1:]
        model_1_policy_gradient_losses += value_loss_1_callback.policy_gradient_losses[1:]

    return model_0_value_losses, model_0_policy_gradient_losses, model_1_value_losses, model_1_policy_gradient_losses


def save_and_load_model_tuples(model_tuples: list[tuple[ModelWrapper, ModelWrapper]]) -> list[tuple[ModelWrapper, ModelWrapper]]:
    copied_model_tuples = []
    time_string = time.strftime('%Y%m%d-%H%M%S')

    for model_tuple in model_tuples:
        model_tuple[0].model.save(f"weights_from_main_with_perturbations/{model_tuple[0].name}_{time_string}")
        copied_model_0 = MaskablePPO.load(f"weights_from_main_with_perturbations/{model_tuple[0].name}_{time_string}")
        copied_model_wrapper_0 = ModelWrapper(model_tuple[0].name, copied_model_0)

        model_tuple[1].model.save(f"weights_from_main_with_perturbations/{model_tuple[1].name}_{time_string}")
        copied_model_1 = MaskablePPO.load(f"weights_from_main_with_perturbations/{model_tuple[1].name}_{time_string}")
        copied_model_wrapper_1 = ModelWrapper(model_tuple[1].name, copied_model_1)

        copied_model_tuples.append((copied_model_wrapper_0, copied_model_wrapper_1))

    print(f"Saved model tuples. Time: {time_string}")

    return copied_model_tuples


def evaluate_model_wrapper_tuple_against_random_agent(env_fn, model_wrapper_tuple: tuple[ModelWrapper, ModelWrapper], num_games, **env_kwargs):
    env = env_fn.env(**env_kwargs)

    print(f"Starting evaluation of {model_wrapper_tuple[0].name} and {model_wrapper_tuple[1].name} vs. random agent")

    scores = {"stationary_model": 0, "training_model": 0}
    total_rewards = {"stationary_model": 0, "training_model": 0}
    round_rewards = []
    game_lengths = []
    random_agent_is_first_player = False

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        random_agent_is_first_player = not random_agent_is_first_player
        game_length = 0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winning_player = max(env.rewards, key=env.rewards.get)

                    if (winning_player == env.possible_agents[0] and random_agent_is_first_player) or (winning_player == env.possible_agents[1] and not random_agent_is_first_player):
                        winning_model = "stationary_model"
                    else:
                        winning_model = "training_model"

                    scores[winning_model] += env.rewards[
                        winning_player
                    ]  # only tracks the largest reward (winner of game)

                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    if (a == env.possible_agents[0] and random_agent_is_first_player) or (a == env.possible_agents[1] and not random_agent_is_first_player):
                        total_rewards["stationary_model"] += env.rewards[a]
                    else:
                        total_rewards["training_model"] += env.rewards[a]

                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                game_lengths.append(game_length)
                break
            else:
                if (agent == env.possible_agents[0] and random_agent_is_first_player) or (agent == env.possible_agents[1] and not random_agent_is_first_player):
                        act = env.action_space(agent).sample(action_mask)
                else:
                    training_model_is_first_player = not random_agent_is_first_player

                    if training_model_is_first_player:
                        act = int(
                            model_wrapper_tuple[0].model.predict(
                                observation, action_masks=action_mask, deterministic=False
                            )[0]
                        )
                    else:
                        act = int(
                            model_wrapper_tuple[1].model.predict(
                                observation, action_masks=action_mask, deterministic=False
                            )[0]
                        )
            env.step(act)
            game_length += 1
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores["training_model"] / sum(scores.values())

    average_game_length = sum(game_lengths) / len(game_lengths)

    # print("Rewards by round: ", round_rewards)
    # print("Total rewards (incl. negative rewards): ", total_rewards)
    # print("Final scores: ", scores)
    print("Winrate: ", winrate)
    print("Average game length: ", average_game_length)
    return round_rewards, total_rewards, winrate, scores, average_game_length



def evaluate_model_wrappers_against_each_other(env_fn, models: tuple[ModelWrapper, ModelWrapper], num_games=500, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(**env_kwargs)

    print(f"Starting evaluation of {models[0].name} vs {models[1].name}.")

    scores = {"player_0": 0, "player_1": 0}
    total_rewards = {"player_0": 0, "player_1": 0}
    round_rewards = []
    game_lengths = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        game_length = 0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winning_player = max(env.rewards, key=env.rewards.get)

                    if winning_player == env.possible_agents[0]:
                        winning_model = "player_0"
                    else:
                        winning_model = "player_1"

                    scores[winning_model] += env.rewards[
                        winning_player
                    ]  # only tracks the largest reward (winner of game)

                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    if a == env.possible_agents[0]:
                        total_rewards["player_0"] += env.rewards[a]
                    else:
                        total_rewards["player_1"] += env.rewards[a]

                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                game_lengths.append(game_length)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = int(
                        models[0].model.predict(
                            observation, action_masks=action_mask, deterministic=False
                        )[0]
                    )
                else:
                    act = int(
                        models[1].model.predict(
                            observation, action_masks=action_mask, deterministic=False
                        )[0]
                    )
            env.step(act)
            game_length += 1
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate_player_0 = 0
    else:
        winrate_player_0 = scores["player_0"] / sum(scores.values())

    average_game_length = sum(game_lengths) / len(game_lengths)

    # print("Rewards by round: ", round_rewards)
    # print("Total rewards (incl. negative rewards): ", total_rewards)
    # print("Final scores: ", scores)
    print("Winrate of player_0: ", winrate_player_0)
    print("Average game length: ", average_game_length)
    return winrate_player_0, average_game_length


def initialize_model_wrapper_tuples(population_size: int) -> list[tuple[ModelWrapper, ModelWrapper]]:
    model_wrapper_tuples = []

    for i in range(population_size):
        env_0 = env_fn.env(**env_kwargs)
        env_0 = SB3ActionMaskWrapper(env_0, None, True)
        env_0.reset(seed=0)
        env_0 = ActionMasker(env_0, mask_fn)
        training_model_0 = MaskablePPO(MaskableActorCriticPolicy, env_0, verbose=1, learning_rate=0.00001)
        training_model_0.set_random_seed(0)
        model_wrapper_0 = ModelWrapper(f"model_{i}_0", training_model_0)

        env_1 = env_fn.env(**env_kwargs)
        env_1 = SB3ActionMaskWrapper(env_1, None, False)
        env_1.reset(seed=0)
        env_1 = ActionMasker(env_1, mask_fn)
        training_model_1 = MaskablePPO(MaskableActorCriticPolicy, env_1, verbose=1, learning_rate=0.00001)
        training_model_1.set_random_seed(0)
        model_wrapper_1 = ModelWrapper(f"model_{i}_1", training_model_1)

        model_wrapper_tuples.append((model_wrapper_0, model_wrapper_1))

    return model_wrapper_tuples


def initialize_model_wrapper_tuples_from_files(model_file_paths: list[tuple[str, str]], names: list[tuple[str, str]]) -> list[tuple[ModelWrapper, ModelWrapper]]:
    if len(model_file_paths) != len(names):
        raise ValueError("Number of model file paths must be the same as number of names")

    model_wrapper_tuples = []

    for i in range(len(model_file_paths)):
        model_0_file_path = model_file_paths[i][0]
        model_0_name = names[i][0]
        model_0_custom_objects = {'learning_rate': 0.00001}
        model_0 = MaskablePPO.load(model_0_file_path, custom_objects=model_0_custom_objects)
        model_wrapper_0 = ModelWrapper(model_0_name, model_0)

        model_1_file_path = model_file_paths[i][1]
        model_1_name = names[i][1]
        model_1_custom_objects = {'learning_rate': 0.00001}
        model_1 = MaskablePPO.load(model_1_file_path, custom_objects=model_1_custom_objects)
        model_wrapper_1 = ModelWrapper(model_1_name, model_1)

        model_wrapper_tuples.append((model_wrapper_0, model_wrapper_1))

    return model_wrapper_tuples


def execute_self_play_training_loop(
    number_of_iterations: int,
    number_of_steps_per_iteration: int,
    population_size: int
):
    # Initialize models
    model_wrapper_tuples = initialize_model_wrapper_tuples(population_size)
    #model_wrapper_tuples = initialize_model_wrapper_tuples_from_files(
    #    [
    #        (
    #            "models_for_initialization/ppo_model_random_1000000_0_00001.zip",
    #            "models_for_initialization/ppo_model_random_1000000_0_00001.zip"
    #        ),
    #        (
    #            "models_for_initialization/ppo_model_random_1000000_0_00001.zip",
    #            "models_for_initialization/ppo_model_random_1000000_0_00001.zip"
    #        )
    #    ],
    #    [("model_0_0", "model_0_1"), ("model_1_0", "model_1_1")],
    #)

    training_progress_data = []

    for i in range(number_of_iterations):
        print("Iteration", i, "of", number_of_iterations)

        # Train models
        player_0_value_losses, player_0_policy_gradient_losses, player_1_value_losses, player_1_policy_gradient_losses = train_action_mask(
            env_fn,
            model_wrapper_tuples,
            steps=number_of_steps_per_iteration,
            seed=0,
            **env_kwargs
        )

        # Save and load models
        model_wrapper_tuples = save_and_load_model_tuples(model_wrapper_tuples)

        # Evaluate against random model
        accumulated_winrate = 0
        accumulated_game_length = 0

        for model_wrapper_tuple in model_wrapper_tuples:
            _, _, winrate, _, average_game_length = evaluate_model_wrapper_tuple_against_random_agent(
                env_fn,
                model_wrapper_tuple,
                num_games=500,
                **env_kwargs
            )

            accumulated_winrate += winrate
            accumulated_game_length += average_game_length

        winrate_random = accumulated_winrate / len(model_wrapper_tuples)
        average_game_length_random = accumulated_game_length / len(model_wrapper_tuples)

        training_progress_data.append({
            "winrate_random": round(winrate_random, 3),
            "average_game_length_random": average_game_length_random,
            "player_0_value_losses": [np.round(value_loss, 3).item() for value_loss in player_0_value_losses],
            "player_0_policy_gradient_losses": [np.round(policy_gradient_loss, 4).item() for policy_gradient_loss in
                                                player_0_policy_gradient_losses],
            "player_1_value_losses": [np.round(value_loss, 3).item() for value_loss in player_1_value_losses],
            "player_1_policy_gradient_losses": [np.round(policy_gradient_loss, 4).item() for policy_gradient_loss in
                                                player_1_policy_gradient_losses],
            "time_stamp": time.strftime('%Y%m%d-%H%M%S')
        })

    save_training_progress_data(training_progress_data)


def save_training_progress_data(training_progress_data: list[dict]):
    history_file_name = f"histories/history_with_perturbations_{time.strftime('%Y%m%d-%H%M%S')}.json"

    with open(history_file_name, "w") as fp:
        json.dump(training_progress_data, fp)

    print("history saved to " + history_file_name)


if __name__ == "__main__":
    if gym.__version__ > "0.29.1":
        raise ImportError(
            f"This script requires gymnasium version 0.29.1 or lower, but you have version {gym.__version__}."
        )

    env_fn = connect_four_v3

    env_kwargs = {}

    # evaluate_model_against_other_models("connect_four_v3_20250207-182953.zip", [None])

    iterations_count = 100
    step_count_per_iteration = 20480
    size_of_population = 2

    execute_self_play_training_loop(
        iterations_count,
        step_count_per_iteration,
        size_of_population
    )
