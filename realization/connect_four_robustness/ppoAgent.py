from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import pettingzoo.utils
from pettingzoo.classic import connect_four_v3
import time
from agent import Agent


class PpoAgent(Agent):
    def __init__(self, name, parameters_file_path=None, seed=0):
        super().__init__(name)
        self.parameters_file_path = parameters_file_path

        if self.parameters_file_path is None:
            env_fn = connect_four_v3
            env_kwargs = {}
            env = env_fn.env(**env_kwargs)

            # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
            env = SB3ActionMaskWrapper(env)

            env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

            env = ActionMasker(env, self._mask_fn)
            self.model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
        else:
            self.model = MaskablePPO.load(parameters_file_path)

    def train(self):
        env_fn = connect_four_v3
        env_kwargs = {}
        self._train_action_mask(env_fn, steps=40_960, seed=0, **env_kwargs)

    def determine_action(self, observation: dict) -> int:
        action_mask = observation["action_mask"]

        chosen_action = int(
            self.model.predict(
                observation["observation"], action_masks=action_mask, deterministic=True
            )[0]
        )
        return chosen_action

    def _mask_fn(self, env):
        # Do whatever you'd like in this function to return the action mask
        # for the current env. In this example, we assume the env has a
        # helpful method we can rely on.
        return env.action_mask()

    def _train_action_mask(self, env_fn, steps=10_000, seed=0, **env_kwargs):
        """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
        env = env_fn.env(**env_kwargs)

        print(f"Starting training on {str(env.metadata['name'])}.")

        # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
        env = SB3ActionMaskWrapper(env)

        env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

        env = ActionMasker(env, self._mask_fn)  # Wrap to enable masking (SB3 function)
        # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
        # with ActionMasker. If the wrapper is detected, the masks are automatically
        # retrieved and used when learning. Note that MaskablePPO does not accept
        # a new action_mask_fn kwarg, as it did in an earlier draft.
        self.model.set_random_seed(seed)
        self.model.learn(total_timesteps=steps)

        self.parameters_file_path = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.model.save(self.parameters_file_path)

        print("Model has been saved.")

        print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

        env.close()


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

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

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info.

        The observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).
        """
        current_agent = self.agent_selection

        super().step(action)

        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]
