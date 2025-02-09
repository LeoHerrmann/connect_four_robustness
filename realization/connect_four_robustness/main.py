from datetime import datetime
import os
import pickle
import custom_connect_four_v3
import matplotlib
import matplotlib.pyplot as plt

from agent import Agent
from mctsAgent import MctsAgent
from ppoAgent import PpoAgent
from randomAgent import RandomAgent
from humanAgent import HumanAgent

observation_history = []

def play_game(env, agents: list[Agent]):
        game_length = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            observation_history.append(observation)

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
                if agent == env.possible_agents[0]:
                    action = agents[0].determine_action(observation)
                else:
                    action = agents[1].determine_action(observation)

            env.step(action)


def play_games(number_of_games, agents: list[Agent], alternate_player_order=True):
    absolute_history = []
    average_history = []
    player_0_win_count = 0
    player_1_win_count = 0
    draw_count = 0
    average_game_length = 0

    for i in range(number_of_games):
        game_options = {
            "reverse_order": False
        }

        if i % 2 == 0 and alternate_player_order:
            game_options["reverse_order"] = True

        env = custom_connect_four_v3.env(render_mode="human")
        # env = custom_connect_four_v3.env()
        # env = tictactoe_v3.env(render_mode="human")
        env.reset(options=game_options)

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

        env.close()

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


def save_absolute_history(absolute_history):
    datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs("results", exist_ok=True)

    with open("results/" + datetime_string + "_absolute_history" + ".pkl", "wb+") as f:
        pickle.dump(absolute_history, f)


def save_average_history_and_figures(average_history, win_rates_figure, game_length_figure):
    datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs("results", exist_ok=True)

    with open("results/" + datetime_string + "_average_history" + ".pkl", "wb+") as f:
        pickle.dump(average_history, f)

    win_rates_figure.savefig(
        "results/" + datetime_string + "_graph_win_rates",
        bbox_inches='tight',
        pad_inches=0
    )

    game_length_figure.savefig(
        "results/" + datetime_string + "_graph_game_length",
        bbox_inches='tight',
        pad_inches=0
    )

    plt.show()



#ppoAgent = PpoAgent("PPO1")
#ppoAgent.train()

number_of_games = 100
alternate_player_order = False

agents = [HumanAgent("HA1"), MctsAgent("HA1", False, n_simulations=10000)]

# agents = [MyMctsAgent("RA1", True, n_simulations=15000), RandomAgent("MA1"),]
# agents = [RandomAgent("MA1"), MyMctsAgent("RA1", False, n_simulations=15000)]
# agents = [MyMctsAgent("MA1", True, n_simulations=15000), MyMctsAgent("RA1", False, n_simulations=15000)]

# agents = [MyMctsAgent("RA1", True, n_simulations=20000), RandomAgent("MA1"),]
# agents = [RandomAgent("MA1"), MyMctsAgent("RA1", False, n_simulations=20000)]
# agents = [MyMctsAgent("MA1", True, n_simulations=20000), MyMctsAgent("RA1", False, n_simulations=20000)]


absolute_history, average_history = play_games(number_of_games, agents, alternate_player_order)
win_rates_figure, game_length_figure = generate_figures(average_history)
save_absolute_history(absolute_history)
save_average_history_and_figures(average_history, win_rates_figure, game_length_figure)
print(average_history[len(average_history) - 1])
