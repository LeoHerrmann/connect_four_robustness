import os
import json
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

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
    plt.ylabel("Gewinnrate / %")
    plt.xlabel("Anzahl der Spiele")
    plt.grid(True)
    plt.legend()

    game_length_figure = plt.figure(2)
    plt.plot(game_indices, average_game_lengths, label="Average Game Length")
    plt.ylabel("Durchschn. Spieldauer / Züge")
    plt.xlabel("Anzahl der Spiele")
    plt.grid(True)

    plt.grid(True)

    return win_rates_figure, game_length_figure


def save_average_history_and_figures(win_rates_figure, game_length_figure, results_subfolder: str, timestamp: str):
    results_subfolder_path = results_subfolder + "/"
    os.makedirs(results_subfolder_path, exist_ok=True)

    win_rates_figure.savefig(
        results_subfolder_path + timestamp + "_graph_win_rates",
        bbox_inches='tight',
        pad_inches=0
    )

    game_length_figure.savefig(
        results_subfolder_path + timestamp + "_graph_game_length",
        bbox_inches='tight',
        pad_inches=0
    )

    plt.show()


read_file_path = "random_vs_random_0/alternating_player_order/20250223-202221_average_history.json"
timestamp = "20250223-202221"
results_subfolder = "random_vs_random_0/alternating_player_order"

average_history = []

with open(read_file_path) as read_file:
	d = json.load(read_file)
	average_history = d["average_history"]

win_rates_figure, game_length_figure = generate_figures(average_history)
save_average_history_and_figures(win_rates_figure, game_length_figure, results_subfolder, timestamp)
