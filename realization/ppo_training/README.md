# PPO Training

This directory contains various Python scripts to train a PPO (Proximal Policy Optimization) agent for Connect Four using different approaches.

## Contents

`histories/`
- Target directory for storing training metrics

`models_for_initialization/`
- Directory for manually storing model weights that can be used for further training.

`weights_from_main_*_/`
- Target directory for storing training weights.

`main_0_from_pettingszoo_docs.py`
- Implementation based on the [a snippet from the PettingZoo documentation](https://pettingzoo.farama.org/tutorials/sb3/connect_four/).

`main_1_with_frozen_agents.py`
- Extends `main_0_from_pettingszoo_docs.py` by ensuring that the training model does not control both agents.
- Instead, it always plays against the latest version of itself that has met a certain win rate threshold in evaluation.

`main_2_with_frozen_and_cached_agents.py`
- Further extends `main_1_with_frozen_agents.py` by changing the opponent selection strategy.
- Instead of playing against only the latest version, the training model competes against a randomly selected agent from a pool of older versions.

`main_3_zhong_et_al.py`
- Implementation of the approach described by [Zhong et al.](https://arxiv.org/abs/2009.06086)

`main_4_random.py`
- Due to time constraints, the previously explored methods could not be fully optimized to develop a strong strategy, so this script was written which implements training against a randomly playing agent.
