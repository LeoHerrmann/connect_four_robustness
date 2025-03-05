import json
import matplotlib
import matplotlib.pyplot as plt

file_name_start = "frozen_and_cached_fixed_0_00001_0_95/history_frozen_and_cached_20250216-035831"

history = []

with open(file_name_start + ".json") as f:
	history = json.load(f)
	f.close()

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.dpi"] = 300

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)

win_rates_stationary = [(item["winrate_stationary"] * 100) for item in history]
win_rates_random = [(item["winrate_random"] * 100) for item in history]
average_game_lengths_stationary = [item["average_game_length_stationary"] for item in history]
average_game_lengths_random = [item["average_game_length_random"] for item in history]
value_losses = [item["value_losses"][1] for item in history]
policy_gradient_losses = [item["policy_gradient_losses"][1] for item in history]
game_indices = range(1, len(history) + 1)

win_rates_figure = plt.figure(1)
plt.plot(game_indices, win_rates_stationary, label="Stationär")
plt.plot(game_indices, win_rates_random, label="Zufällig")
plt.ylabel("Gewinnrate [%]")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)
plt.legend()

game_lengths_figure = plt.figure(2)
plt.plot(game_indices, average_game_lengths_stationary, label="Stationär")
plt.plot(game_indices, average_game_lengths_random, label="Zufällig")
plt.ylabel("Durchschnittliche Spieldauer")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)
plt.legend()

value_losses_figure = plt.figure(3)
plt.plot(game_indices, value_losses, label="Werteverlust")
plt.ylabel("Werteverlust")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)

policy_gradient_losses_figure = plt.figure(4)
plt.plot(game_indices, policy_gradient_losses, label="Policy-Gradient-Verlust")
plt.ylabel("Policy-Gradient-Verlust")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)


win_rates_figure.savefig(
    file_name_start + "_graph_win_rates",
    bbox_inches='tight',
    pad_inches=0
)

game_lengths_figure.savefig(
    file_name_start + "_graph_game_lengths",
    bbox_inches='tight',
    pad_inches=0
)

value_losses_figure.savefig(
    file_name_start + "_graph_value_losses",
    bbox_inches='tight',
    pad_inches=0
)

policy_gradient_losses_figure.savefig(
    file_name_start + "_graph_policy_gradient_losses",
    bbox_inches='tight',
    pad_inches=0
)
