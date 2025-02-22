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

    def __init__(self, env, training_agent_is_player_0: bool):
        super().__init__(env)
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
            stationary_agent_action_mask = self.action_mask()
            stationary_agent_legal_actions = [i for i, is_legal in enumerate(stationary_agent_action_mask) if is_legal]
            stationary_agent_chosen_action = random.choice(stationary_agent_legal_actions)
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

        # Execute step for frozen player randomly
        stationary_agent_action_mask = self.action_mask()
        stationary_agent_legal_actions = [i for i, is_legal in enumerate(stationary_agent_action_mask) if is_legal]
        stationary_agent_chosen_action = random.choice(stationary_agent_legal_actions)

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
        self.mean_episode_lengths = []
        self.mean_episode_rewards = []
        self.value_losses = []
        self.policy_gradient_losses = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.value_losses.append(self.logger.name_to_value["train/value_loss"])
        self.policy_gradient_losses.append(self.logger.name_to_value["train/policy_gradient_loss"])
        #self.mean_episode_lengths.append(self.logger.name_to_value["rollout/ep_len_mean"])
        #self.mean_episode_rewards.append(self.logger.name_to_value["rollout/ep_rew_mean"])


def train_action_mask(env_fn, model: MaskablePPO, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""

    value_losses = []
    policy_gradient_losses = []
    mean_episode_lengths = []
    mean_episode_rewards = []

    env_0 = env_fn.env(**env_kwargs)
    print(f"Starting training of PPO model against random agent.")
    env_0 = SB3ActionMaskWrapper(env_0, True)
    env_0.reset(seed=seed)  # Must call reset() in order to re-define the spaces
    env_0 = ActionMasker(env_0, mask_fn)  # Wrap to enable masking (SB3 function)
    model.set_env(env_0)
    value_loss_0_callback = ValueLossCallback()
    model.learn(total_timesteps=steps, callback=value_loss_0_callback, reset_num_timesteps=False)
    model.save(f"weights_from_main_against_random/ppo_model_{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"Model has been saved.")
    print(f"Finished training on {str(env_0.unwrapped.metadata['name'])}.\n")

    env_0.close()
    value_losses += value_loss_0_callback.value_losses[1:]
    policy_gradient_losses += value_loss_0_callback.policy_gradient_losses[1:]
    mean_episode_lengths += value_loss_0_callback.mean_episode_lengths[1:]
    mean_episode_rewards += value_loss_0_callback.mean_episode_rewards[1:]

    return value_losses, policy_gradient_losses


def evaluate_model_wrapper_tuple_against_random_agent(env_fn, model: MaskablePPO, num_games, **env_kwargs):
    env = env_fn.env(**env_kwargs)

    print(f"Starting evaluation of model vs. random agent")

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
                    act = int(
                        model.predict(
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


def execute_training(number_of_steps_per_iteration: int, learning_rate: float):
    # Initialize model
    env = env_fn.env(**env_kwargs)
    env = SB3ActionMaskWrapper(env, True)
    env.reset(seed=0)
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, learning_rate=learning_rate)
    model.set_random_seed(0)

    training_progress_data = []

    # Train model
    value_losses, policy_gradient_losses = train_action_mask(
        env_fn,
        model,
        steps=number_of_steps_per_iteration,
        seed=0,
        **env_kwargs
    )

    # Evaluate against random model

    _, _, winrate, _, average_game_length = evaluate_model_wrapper_tuple_against_random_agent(
        env_fn,
        model,
        num_games=1000,
        **env_kwargs
    )

    training_progress_data.append({
        "winrate_random_final": round(winrate, 3),
        "average_game_length_random_final": average_game_length,
        #"mean_episode_lengths": [np.round(mean_episode_length, 3).item() for mean_episode_length in mean_episode_lengths],
        #"mean_episode_rewards": [np.round(mean_episode_reward, 3).item() for mean_episode_reward in mean_episode_rewards],
        "value_losses": [np.round(value_loss, 3).item() for value_loss in value_losses],
        "policy_gradient_losses": [np.round(policy_gradient_loss, 4).item() for policy_gradient_loss in policy_gradient_losses],
        "time_stamp": time.strftime('%Y%m%d-%H%M%S')
    })

    save_training_progress_data(training_progress_data)


def save_training_progress_data(training_progress_data: list[dict]):
    history_file_name = f"histories/history_against_random_{time.strftime('%Y%m%d-%H%M%S')}.json"

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

    step_count_per_iteration = 1_024_000
    learning_rate = 0.003 # 0.001, 0.0005, 0.0001

    execute_training(
        step_count_per_iteration,
        learning_rate
    )
