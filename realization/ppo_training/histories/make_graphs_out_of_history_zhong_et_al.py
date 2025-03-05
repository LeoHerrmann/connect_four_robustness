import json
import statistics
import matplotlib
import matplotlib.pyplot as plt

file_name_start = "perturbations/10_policy_updates/0_00003/history_with_perturbations_20250228-035928"

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

win_rates_random = [(item["winrate_random"] * 100) for item in history]
average_game_lengths_random = [item["average_game_length_random"] for item in history]
player_0_value_losses = [statistics.mean(item["player_0_value_losses"]) for item in history]
player_1_value_losses = [statistics.mean(item["player_1_value_losses"]) for item in history]
player_0_policy_gradient_losses = [statistics.mean(item["player_0_policy_gradient_losses"]) for item in history]
player_1_policy_gradient_losses = [statistics.mean(item["player_1_policy_gradient_losses"]) for item in history]
game_indices = range(1, len(history) + 1)

win_rates_figure = plt.figure(1)
plt.ylim(50, 100)
plt.plot(game_indices, win_rates_random)
plt.ylabel("Gewinnrate / %")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)

game_lengths_figure = plt.figure(2)
plt.ylim(10, 22)
plt.plot(game_indices, average_game_lengths_random)
plt.ylabel("Durchschn. Spieldauer / Züge")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)

value_losses_figure = plt.figure(3)
plt.ylim(0, 0.7)
plt.plot(game_indices, player_0_value_losses, label="Spieler 0")
plt.plot(game_indices, player_1_value_losses, label="Spieler 1")
plt.ylabel("Werteverlust")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)
plt.legend()

policy_gradient_losses_figure = plt.figure(4)
plt.ylim(-0.05, 0)
plt.plot(game_indices, player_0_policy_gradient_losses, label="Spieler 0")
plt.plot(game_indices, player_1_policy_gradient_losses, label="Spieler 1")
plt.ylabel("Policy-Gradient-Verlust")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)
plt.legend()


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
