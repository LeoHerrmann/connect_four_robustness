from datetime import datetime
import os
import json
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

from pettingzoo.classic import connect_four_v3
from distortionGenerator import DistortionGenerator

from agent import Agent
from mctsAgent import MctsAgent
from ppoAgent import PpoAgent
from randomAgent import RandomAgent
from humanAgent import HumanAgent


def play_game(env, agents: list[Agent]):
    game_length = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        game_length += 1

        if termination or truncation:
            statistics = {
                "result": "",
                "game_length": game_length
            }

            if env.rewards[env.possible_agents[0]] != env.rewards[env.possible_agents[1]]:
                winner = max(env.rewards, key=env.rewards.get)

                if winner == env.agents[0]:
                    statistics["result"] = "player_0"
                elif winner == env.agents[1]:
                    statistics["result"] = "player_1"

            else:
                statistics["result"] = "draw"
            print(statistics)
            return statistics

        else:
            distorted_state = distortion_generator.distort_state(observation["observation"])
            distorted_observation = observation
            distorted_observation["observation"] = distorted_state

            if agent == env.possible_agents[0]:
                action = agents[0].determine_action(observation)
            else:
                action = agents[1].determine_action(observation)

            distorted_action = distortion_generator.distort_action(action, observation["action_mask"])

        env.step(distorted_action)


def play_games(number_of_games, agents: list[Agent]):
    absolute_history = []
    average_history = []
    player_0_win_count = 0
    player_1_win_count = 0
    draw_count = 0
    average_game_length = 0

    for i in range(number_of_games):
        # env = connect_four_v3.env(render_mode="human")
        env = connect_four_v3.env()
        env.reset()

        game_statistics = play_game(env, agents)

        if game_statistics["result"] == "player_0":
            player_0_win_count += 1
        elif game_statistics["result"] == "player_1":
            player_1_win_count += 1
        elif game_statistics["result"] == "draw":
            draw_count += 1

        player_0_win_rate = player_0_win_count / (len(average_history) + 1)
        player_1_win_rate = player_1_win_count / (len(average_history) + 1)
        average_game_length = (game_statistics["game_length"] - average_game_length) / (len(average_history) + 1) + average_game_length

        absolute_history.append({
            "winner": game_statistics["result"],
            "game_length": game_statistics["game_length"]
        })

        average_history.append({
            "player_0_win_rate": player_0_win_rate,
            "player_1_win_rate": player_1_win_rate,
            "average_game_length": average_game_length
        })

        print(average_history[len(average_history) - 1])

        # env.close()

    return absolute_history, average_history


def generate_figures(average_history):
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = 300

    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    player_0_win_rates = [(item["player_0_win_rate"] * 100) for item in average_history]
    player_1_win_rates = [(item["player_1_win_rate"] * 100) for item in average_history]
    average_game_lengths = [item["average_game_length"] for item in average_history]
    game_indices = range(1, len(average_history) + 1)

    win_rates_figure = plt.figure(1)
    plt.plot(game_indices, player_0_win_rates, label="Spieler 0")
    plt.plot(game_indices, player_1_win_rates, label="Spieler 1")
    plt.ylabel("Gewinnrate [%]")
    plt.xlabel("Anzahl der Spiele")
    plt.grid(True)
    plt.legend()

    game_length_figure = plt.figure(2)
    plt.plot(game_indices, average_game_lengths, color="black", label="Average Game Length")
    plt.ylabel("Durchschnittliche Spieldauer")
    plt.xlabel("Anzahl der Spiele")
    plt.grid(True)

    plt.grid(True)

    return win_rates_figure, game_length_figure


def save_absolute_history(absolute_history: list[dict], results_subfolder: str):
    datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")

    results_subfolder_path = "results/" + results_subfolder + "/"
    os.makedirs(results_subfolder_path, exist_ok=True)

    with open(results_subfolder_path + datetime_string + "_absolute_history" + ".json", "w+") as f:
        json.dump({"absolute_history": absolute_history}, f)


def save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder: str):
    datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")

    results_subfolder_path = "results/" + results_subfolder + "/"
    os.makedirs(results_subfolder_path, exist_ok=True)

    with open(results_subfolder_path + datetime_string + "_average_history.json", "w+") as f:
        json.dump({"average_history": average_history}, f)

    win_rates_figure.savefig(
        results_subfolder_path + datetime_string + "_graph_win_rates",
        bbox_inches='tight',
        pad_inches=0
    )

    game_length_figure.savefig(
        results_subfolder_path + datetime_string + "_graph_game_length",
        bbox_inches='tight',
        pad_inches=0
    )

    plt.show(block=False)


distortion_generator = DistortionGenerator(0, 0.0)

# Evaluate MCTS vs. MCTS

number_of_games = 200

numbers_of_mcts_simulations = [50]    # [50, 100, 250, 500, 750, 1000, 2500]

for number_of_mcts_simulations in numbers_of_mcts_simulations:
    agents = [MctsAgent("MA1", True, number_of_mcts_simulations), MctsAgent("MA1", False, number_of_mcts_simulations)]
    results_subfolder = "mcts_vs_mcts_" + str(number_of_mcts_simulations)

    absolute_history, average_history = play_games(number_of_games, agents)
    win_rates_figure, game_length_figure = generate_figures(average_history)
    save_absolute_history(absolute_history, results_subfolder)
    save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)
    print(average_history[len(average_history) - 1])

exit()

# Evaluate PPO vs. Random

number_of_games = 10000
learning_rates = ["0_003", "0_001", "0_0005", "0_0001"]

for learning_rate in learning_rates:
    print(f"Starting PPO vs. Random with learning rate {learning_rate}")
	
    agents = [PpoAgent("PA1", f"ppoWeights/ppo_model_random_1000000_{learning_rate}"), RandomAgent("RA1")]
    results_subfolder = f"ppo_quantitative_ppo_vs_random_constant_player_order/ppo_vs_random_{learning_rate}"

    absolute_history, average_history = play_games(number_of_games, agents)
    win_rates_figure, game_length_figure = generate_figures(average_history)
    save_absolute_history(absolute_history, results_subfolder)
    save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)

    print(f"Starting Random vs. PPO with learning rate {learning_rate}")

    agents = [RandomAgent("RA1"), PpoAgent("PA1", f"ppoWeights/ppo_model_random_1000000_{learning_rate}")]
    results_subfolder = f"ppo_quantitative_random_vs_ppo_constant_player_order/random_vs_ppo_{learning_rate}"

    absolute_history, average_history = play_games(number_of_games, agents)
    win_rates_figure, game_length_figure = generate_figures(average_history)
    save_absolute_history(absolute_history, results_subfolder)
    save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)

exit()

# Evaluate MCTS vs. Human

number_of_games = 100

agents = [HumanAgent("HA1"), MctsAgent("MA1", False, 5000)]
absolute_history, average_history = play_games(number_of_games, agents)


# Evaluate PPO vs Random with distortions

number_of_games = 1000

numbers_of_fields_to_distort = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

for number_of_fields_to_distort in numbers_of_fields_to_distort:
    print("STARTING WITH", number_of_fields_to_distort, "FIELDS TO DISTORT")

    distortion_generator = DistortionGenerator(number_of_fields_to_distort, 0.0)

    print("STARTING RANDOM VS PPO")
    results_subfolder = "random_vs_ppo_" + str(number_of_fields_to_distort) + "_0"
    agents = [RandomAgent("RA1"), PpoAgent("PA1", "ppoWeights/ppo_model_random_1000000_0_00001.zip")]

    absolute_history, average_history = play_games(number_of_games, agents)
    win_rates_figure, game_length_figure = generate_figures(average_history)
    save_absolute_history(absolute_history, results_subfolder)
    save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)
    print(average_history[len(average_history) - 1])

    print("STARTING PPO VS RANDOM")
    results_subfolder = "ppo_vs_random_" + str(number_of_fields_to_distort) + "_0"
    agents = [PpoAgent("PA1", "ppoWeights/ppo_model_random_1000000_0_00001.zip"), RandomAgent("RA1")]

    absolute_history, average_history = play_games(number_of_games, agents)
    win_rates_figure, game_length_figure = generate_figures(average_history)
    save_absolute_history(absolute_history, results_subfolder)
    save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)
    print(average_history[len(average_history) - 1])

probabilities_of_distorting_actions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for probability_of_distorting_actions in probabilities_of_distorting_actions:
    print("STARTING WITH", probability_of_distorting_actions, "Probability of distorting actions")

    distortion_generator = DistortionGenerator(0, probability_of_distorting_actions)

    print("STARTING RANDOM VS PPO")
    results_subfolder = "random_vs_ppo_" + "0_" + str(probability_of_distorting_actions)
    agents = [RandomAgent("RA1"), PpoAgent("PA1", "ppoWeights/ppo_model_random_1000000_0_00001.zip")]

    absolute_history, average_history = play_games(number_of_games, agents)
    win_rates_figure, game_length_figure = generate_figures(average_history)
    save_absolute_history(absolute_history, results_subfolder)
    save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)
    print(average_history[len(average_history) - 1])

    print("STARTING PPO VS RANDOM")
    results_subfolder = "ppo_vs_random_" + "0_" + str(probability_of_distorting_actions)
    agents = [PpoAgent("PA1", "ppoWeights/ppo_model_random_1000000_0_00001.zip"), RandomAgent("RA1")]

    absolute_history, average_history = play_games(number_of_games, agents)
    win_rates_figure, game_length_figure = generate_figures(average_history)
    save_absolute_history(absolute_history, results_subfolder)
    save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)
    print(average_history[len(average_history) - 1])

# Evaluate MCTS vs Random with distortions

# distortion_generator = DistortionGenerator(0, 0.0)
# number_of_games = 100
# number_of_mcts_simulations = 5000

# print("STARTING RANDOM VS MCTS")
# results_subfolder = "random_vs_mcts_" + str(number_of_mcts_simulations)
# agents = [RandomAgent("RA1"), MctsAgent("MC1", False, n_simulations=number_of_mcts_simulations)]

# absolute_history, average_history = play_games(number_of_games, agents)
# win_rates_figure, game_length_figure = generate_figures(average_history)
# save_absolute_history(absolute_history, results_subfolder)
# save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)
# print(average_history[len(average_history) - 1])

# print("STARTING MCTS VS RANDOM")
# results_subfolder = "mcts_vs_random_" + str(number_of_mcts_simulations)
# agents = [MctsAgent("MC1", True, n_simulations=number_of_mcts_simulations), RandomAgent("RA1")]

# absolute_history, average_history = play_games(number_of_games, agents)
# win_rates_figure, game_length_figure = generate_figures(average_history)
# save_absolute_history(absolute_history, results_subfolder)
# save_average_history_and_figures(average_history, win_rates_figure, game_length_figure, results_subfolder)
# print(average_history[len(average_history) - 1])
