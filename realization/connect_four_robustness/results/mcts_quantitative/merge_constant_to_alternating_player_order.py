import json
import matplotlib
matplotlib.use('gtk3agg') 
import matplotlib.pyplot as plt

simulation_count = str(5000)

path_to_absolute_history_mcts_vs_random = "mcts_vs_random_constant_player_order/mcts_vs_random_" + simulation_count + "/absolute_history.json"
path_to_absolute_history_random_vs_mcts = "random_vs_mcts_constant_player_order/random_vs_mcts_" + simulation_count + "/absolute_history.json"
destination_path_for_average_history = "mcts_vs_random_alternating_player_order/" + simulation_count + "/average_history.json"
destination_path_for_absolute_history = "mcts_vs_random_alternating_player_order/" + simulation_count + "/absolute_history.json"
destination_path_for_average_win_rate_graph = "mcts_vs_random_alternating_player_order/" + simulation_count + "/win_rate.png"
destination_path_for_average_game_length_graph = "mcts_vs_random_alternating_player_order/" + simulation_count + "/game_length.png"


def generate_average_history_figures(average_history: list[dict]):
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = 300

    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    player_0_win_rates = [(item["win_rate_mcts"] * 100) for item in average_history]
    player_1_win_rates = [(item["win_rate_random"] * 100) for item in average_history]
    average_game_lengths = [item["average_game_length"] for item in average_history]
    game_indices = range(1, len(average_history) + 1)

    win_rates_figure = plt.figure(1)
    plt.plot(game_indices, player_0_win_rates, label="MCTS")
    plt.plot(game_indices, player_1_win_rates, label="Zufällig")
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


def save_figures(win_rates_figure, game_length_figure):
    win_rates_figure.savefig(
        destination_path_for_average_win_rate_graph,
        bbox_inches='tight',
        pad_inches=0
    )

    game_length_figure.savefig(
        destination_path_for_average_game_length_graph,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.show()


# Die vier JSON JSON-Dateien öffnen und die Listen da raus holen
average_history_mcts_vs_random = []
average_history_random_vs_mcts = []
absolute_history_mcts_vs_random = []
absolute_history_random_vs_mcts = []

with open(path_to_absolute_history_mcts_vs_random) as f:
    absolute_history_mcts_vs_random = json.load(f)["absolute_history"]

with open(path_to_absolute_history_random_vs_mcts) as f:
    absolute_history_random_vs_mcts = json.load(f)["absolute_history"]


# Sicherstellen, dass die geladenen absoluten Historien gleich lang sind
if len(absolute_history_mcts_vs_random) != len(absolute_history_random_vs_mcts):
	exit("Die Historien sind nicht gleich lang. Die Historien müssen gleich lang sein :(")

# Zusammengefügte absolute Historie berechnen
merged_absolute_history = []

for i in range(len(absolute_history_mcts_vs_random)):
	new_history_item_1 = {
		"winner": "MCTS" if absolute_history_mcts_vs_random[i]["winner"] == "player_0" else "Zufällig",
		"game_length": absolute_history_mcts_vs_random[i]["game_length"]
	}
	
	new_history_item_2 = {
		"winner": "MCTS" if absolute_history_random_vs_mcts[i]["winner"] == "player_1" else "Zufällig",
		"game_length": absolute_history_random_vs_mcts[i]["game_length"]
	}

	merged_absolute_history.append(new_history_item_1)
	merged_absolute_history.append(new_history_item_2)

# Zusammengeführte durchschnittliche Historie berechnen
merged_average_history = []

total_wins_mcts = 0
total_wins_random = 0
total_game_length = 0

for i in range(len(merged_absolute_history)):
	game_count = i + 1
	
	total_game_length += merged_absolute_history[i]["game_length"]
	
	if merged_absolute_history[i]["winner"] == "MCTS":
		total_wins_mcts += 1
	elif merged_absolute_history[i]["winner"] == "Zufällig":
		total_wins_random += 1
	
	new_history_item = {
		"win_rate_mcts": total_wins_mcts / game_count,
		"win_rate_random": total_wins_random / game_count,
		"average_game_length": total_game_length / game_count
	}
	
	merged_average_history.append(new_history_item)

# In JSON abspeichern
with open(destination_path_for_average_history, "w+") as f:
    json.dump({"average_history": merged_average_history}, f)

with open(destination_path_for_absolute_history, "w+") as f:
    json.dump({"absolute_history": merged_absolute_history}, f)

# Graphen generieren
average_history_win_rate_figure, average_history_game_length_figure = generate_average_history_figures(merged_average_history)
save_figures(average_history_win_rate_figure, average_history_game_length_figure)

