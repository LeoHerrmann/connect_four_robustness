"""
Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
"""

import glob
import os
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


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def __init__(self, env, stationary_models: list[MaskablePPO | None]):
        super().__init__(env)
        self.stationary_models = stationary_models
        self.stationary_model_selected_for_current_game = None
        self.training_agent_is_player_0 = True

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

        # Toggle player order
        self.training_agent_is_player_0 = not self.training_agent_is_player_0

        # Select a stationary model for the current game
        self.stationary_model_selected_for_current_game = random.choice(self.stationary_models)

        # If training agent is second player, execute first step based on stationary policy
        if not self.training_agent_is_player_0:
            stationary_agent_observation = self.observe(self.agent_selection)
            stationary_agent_action_mask = self.action_mask()

            if self.stationary_model_selected_for_current_game is None:
                # If no stationary models are given execute step based on random policy
                stationary_agent_legal_actions = [i for i, is_legal in enumerate(stationary_agent_action_mask) if is_legal]
                stationary_agent_chosen_action = random.choice(stationary_agent_legal_actions)
            else:
                # If stationary models are given execute step based on a random stationary model
                stationary_agent_chosen_action = int(
                    self.stationary_model_selected_for_current_game.predict(
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

        if self.stationary_model_selected_for_current_game is None:
            # If no stationary models are given execute step based on random policy
            stationary_agent_legal_actions = [i for i, is_legal in enumerate(stationary_agent_action_mask) if is_legal]
            stationary_agent_chosen_action = random.choice(stationary_agent_legal_actions)
        else:
            # If stationary models are given execute step based on given model
            stationary_agent_chosen_action = int(
                self.stationary_model_selected_for_current_game.predict(
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


def train_action_mask(env_fn, steps=10_000, seed=0, stationary_models: list[MaskablePPO | None] = None, training_model: MaskablePPO = None, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env, stationary_models)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    if training_model is None:
        training_model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, learning_rate=0.00001)
        training_model.set_random_seed(seed)

    value_loss_callback = ValueLossCallback()

    training_model.learn(total_timesteps=steps, callback=value_loss_callback)

    training_model.save(f"weights_from_main_with_frozen_and_cached_agents/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()

    return value_loss_callback.value_losses, value_loss_callback.policy_gradient_losses


def eval_action_mask(env_fn, num_games=100, stationary_model_paths: list[str] = None, training_model_path: str = None, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print("Starting evaluation of")

    for stationary_model_path in stationary_model_paths:
        print(stationary_model_path if stationary_model_path is not None else 'random_agent')

    print(f"vs {training_model_path if training_model_path is not None else 'random_agent'}.")

    stationary_model_list = []
    training_model = None

    for stationary_model_path in stationary_model_paths:
        if stationary_model_path is not None:
            stationary_model_list.append(MaskablePPO.load(stationary_model_path))
        else:
            stationary_model_list.append(None)

    if training_model_path is not None:
        training_model = MaskablePPO.load(training_model_path)

    scores = {"stationary_model": 0, "training_model": 0}
    total_rewards = {"stationary_model": 0, "training_model": 0}
    round_rewards = []
    game_lengths = []
    stationary_model_is_first_player = False

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        stationary_model_is_first_player = not stationary_model_is_first_player
        randomly_selected_stationary_model = random.choice(stationary_model_list)
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

                    if (winning_player == env.possible_agents[0] and stationary_model_is_first_player) or (winning_player == env.possible_agents[1] and not stationary_model_is_first_player):
                        winning_model = "stationary_model"
                    else:
                        winning_model = "training_model"

                    scores[winning_model] += env.rewards[
                        winning_player
                    ]  # only tracks the largest reward (winner of game)

                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    if (a == env.possible_agents[0] and stationary_model_is_first_player) or (a == env.possible_agents[1] and not stationary_model_is_first_player):
                        total_rewards["stationary_model"] += env.rewards[a]
                    else:
                        total_rewards["training_model"] += env.rewards[a]

                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                game_lengths.append(game_length)
                break
            else:
                if (agent == env.possible_agents[0] and stationary_model_is_first_player) or (agent == env.possible_agents[1] and not stationary_model_is_first_player):
                    if randomly_selected_stationary_model is None:
                        act = env.action_space(agent).sample(action_mask)
                    else:
                        act = int(
                            randomly_selected_stationary_model.predict(
                                observation, action_masks=action_mask, deterministic=False
                            )[0]
                        )
                else:
                    if training_model is None:
                        act = env.action_space(agent).sample(action_mask)
                    else:
                        act = int(
                            training_model.predict(
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


def evaluate_model_against_other_models(model_path, other_model_paths):
    env_fn = connect_four_v3
    env_kwargs = {}

    for other_model_path in other_model_paths:
        eval_action_mask(
            env_fn,
            num_games=100,
            model_0_path=model_path,
            training_model_path=other_model_path,
            render_mode="human",
            **env_kwargs
        )


def try_get_latest_policy_path():
    try:
        env = env_fn.env(render_mode=None, **env_kwargs)

        latest_policy_path = max(
            glob.glob(f"weights_from_main_with_frozen_and_cached_agents/{env.metadata['name']}*.zip"), key=os.path.getctime
        )

        return latest_policy_path
    except ValueError:
        print("Policy not found.")
        exit(0)


def execute_self_play_training_loop(
    number_of_iterations: int,
    number_of_steps_per_iteration: int,
    threshold_winrate_for_updating_stationary_model: float,
    max_number_of_stationary_models: int
):
    policy_paths = []
    path_to_latest_model_which_made_more_than_winning_threshold = None

    stationary_models = [None for _ in range(max_number_of_stationary_models)]
    stationary_model_paths = [None for _ in range(max_number_of_stationary_models)]
    training_model = None

    training_progress_data = []

    for i in range(number_of_iterations):
        print("Iteration", i, "of", number_of_iterations)

        value_losses, policy_gradient_losses = train_action_mask(
            env_fn,
            steps=number_of_steps_per_iteration,
            seed=0,
            stationary_models=stationary_models,
            training_model=training_model,
            **env_kwargs
        )

        latest_policy_path = try_get_latest_policy_path()
        policy_paths.append(latest_policy_path)

        # Evaluate against latest stationary model todo: use actual models instead of paths for evaluation
        _, _, winrate_stationary, _, average_game_length_stationary = eval_action_mask(
            env_fn,
            num_games=100,
            stationary_model_paths=stationary_model_paths,
            training_model_path=latest_policy_path,
            render_mode=None,
            **env_kwargs
        )

        # Evaluate against random model
        _, _, winrate_random, _, average_game_length_random = eval_action_mask(
            env_fn,
            num_games=100,
            stationary_model_paths=[None],
            training_model_path=latest_policy_path,
            render_mode=None,
            **env_kwargs
        )

        training_progress_data.append({
            "winrate_stationary": round(winrate_stationary, 3),
            "average_game_length_stationary": average_game_length_stationary,
            "winrate_random": round(winrate_random, 3),
            "average_game_length_random": average_game_length_random,
            "value_losses": [np.round(value_loss, 3).item() for value_loss in value_losses],
            "policy_gradient_losses": [np.round(policy_gradient_loss, 4).item() for policy_gradient_loss in policy_gradient_losses],
            "latest_stationary_policy": stationary_model_paths[len(stationary_model_paths) - 1]
        })

        # Create new stationary model and add it to stationary_models
        if winrate_stationary > threshold_winrate_for_updating_stationary_model:
            new_stationary_model = MaskablePPO.load(latest_policy_path)

            if len(stationary_models) >= max_number_of_stationary_models:
                stationary_models.pop(0)

            if len(stationary_model_paths) >= max_number_of_stationary_models:
                stationary_model_paths.pop(0)

            stationary_models.append(new_stationary_model)
            stationary_model_paths.append(latest_policy_path)

        env = env_fn.env(**env_kwargs)
        env = SB3ActionMaskWrapper(env, stationary_models)
        env.reset(seed=0)
        env = ActionMasker(env, mask_fn)

        training_model = MaskablePPO.load(latest_policy_path, env)

    save_training_progress_data(training_progress_data)


def save_training_progress_data(training_progress_data: list[dict]):
    history_file_name = f"histories/history_frozen_and_cached_{time.strftime('%Y%m%d-%H%M%S')}.json"

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

    iterations_count = 1500
    step_count_per_iteration = 4096
    threshold_winrate = 0.60
    max_count_of_stationary_models = 5

    execute_self_play_training_loop(
        iterations_count,
        step_count_per_iteration,
        threshold_winrate,
        max_count_of_stationary_models
    )
