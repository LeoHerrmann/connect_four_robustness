import json
import statistics
import matplotlib
import matplotlib.pyplot as plt

file_name_start = "random/1000000_iterations/more_information/history_random_0_0001_20250223-151546"

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

value_losses = [item for item in history[0]["value_losses"]]
policy_gradient_losses = [item for item in history[0]["policy_gradient_losses"]]
mean_episode_lengths = [item for item in history[0]["mean_episode_lengths"]]
mean_episode_rewards = [item for item in history[0]["mean_episode_rewards"]]
game_indices = [i * 2048 for i in range(1, len(history[0]["value_losses"]) + 1)]

mean_episode_lengths_figure = plt.figure(1)
plt.ylim(4, 12)
plt.plot(game_indices, mean_episode_lengths, label="Spieler 0")
plt.ylabel("Durchschn. Ep.-Länge / Schritte")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)

mean_episode_rewards_figure = plt.figure(2)
plt.ylim(0, 1)
plt.plot(game_indices, mean_episode_rewards, label="Spieler 0")
plt.ylabel("Durchschnittliche Belohnung")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)

value_losses_figure = plt.figure(3)
plt.ylim(0, 0.6)
plt.plot(game_indices, value_losses, label="Spieler 0")
plt.ylabel("Werteverlust")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)

policy_gradient_losses_figure = plt.figure(4)
plt.ylim(-0.08, 0.02)
plt.plot(game_indices, policy_gradient_losses, label="Spieler 0")
plt.ylabel("Policy-Gradient-Verlust")
plt.xlabel("Anzahl der Iterationen")
plt.grid(True)

mean_episode_lengths_figure.savefig(
    file_name_start + "_graph_episode_lengths",
    bbox_inches='tight',
    pad_inches=0.05
)

mean_episode_rewards_figure.savefig(
    file_name_start + "_graph_episode_rewards",
    bbox_inches='tight',
    pad_inches=0.05
)

value_losses_figure.savefig(
    file_name_start + "_graph_value_losses",
    bbox_inches='tight',
    pad_inches=0.05
)

policy_gradient_losses_figure.savefig(
    file_name_start + "_graph_policy_gradient_losses",
    bbox_inches='tight',
    pad_inches=0.05
)
