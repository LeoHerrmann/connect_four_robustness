import json
import matplotlib
matplotlib.use('gtk3agg') 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

destination_path_for_average_win_rate_graph = "ppo_vs_random_alternating_player_order/win_rate_vs_learning_rate.png"
destination_path_for_average_game_length_graph = "ppo_vs_random_alternating_player_order/game_length_vs_learning_rate.png"

def generate_figures(
    learning_rates,
    final_win_rates_ppo,
    final_win_rates_ppo_cis,
    final_win_rates_random,
    final_win_rates_random_cis,
    final_game_lengths,
    final_game_lengths_cis
):
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = 300

    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    final_win_rates_ppo_times_hundred = [win_rate * 100 for win_rate in final_win_rates_ppo]
    final_win_rates_ppo_cis_times_hundred = [(interval[0] * 100, interval[1] * 100) for interval in final_win_rates_ppo_cis]
    final_win_rates_random_times_hundred = [win_rate * 100 for win_rate in final_win_rates_random]
    final_win_rates_random_cis_times_hundred = [(interval[0] * 100, interval[1] * 100) for interval in final_win_rates_random_cis]

    ppo_win_rates_lowers = [interval[0] for interval in final_win_rates_ppo_cis_times_hundred]
    ppo_win_rates_uppers = [interval[1] for interval in final_win_rates_ppo_cis_times_hundred]
    random_win_rates_lowers = [interval[0] for interval in final_win_rates_random_cis_times_hundred]
    random_win_rates_uppers = [interval[1] for interval in final_win_rates_random_cis_times_hundred]

    win_rates_figure = plt.figure(1)
    plt.plot(learning_rates, final_win_rates_ppo_times_hundred, marker="o", label="PPO", color="tab:blue", markersize=3)
    plt.plot(learning_rates, final_win_rates_random_times_hundred, marker="o", label="Zufällig", color="tab:orange", markersize=3)
    plt.xticks(rotation=45, ha='right')
    
    for i in range(len(learning_rates)):
        learning_rate = learning_rates[i]
        plt.plot([learning_rate, learning_rate], [ppo_win_rates_lowers[i], ppo_win_rates_uppers[i]], color="tab:blue", marker="_")
        plt.plot([learning_rate, learning_rate], [random_win_rates_lowers[i], random_win_rates_uppers[i]], color="tab:orange", marker="_")
    
    plt.ylabel("Gewinnrate / %")
    plt.xlabel("Lernrate")
    plt.grid(True)
    plt.legend()

    game_lengths_figure = plt.figure(2)
    plt.plot(learning_rates, final_game_lengths, marker="o", label="Durchschn. Spieldauer [Züge]", markersize=3)
    plt.xticks(rotation=45, ha='right')
    
    for i in range(len(learning_rates)):
        learning_rate = learning_rates[i]
        plt.plot([learning_rate, learning_rate], [final_game_lengths_cis[i][0], final_game_lengths_cis[i][1]], marker="_", color="tab:blue")
    
    plt.ylabel("Durchschn. Spieldauer / Züge")
    plt.xlabel("Lernrate")
    plt.grid(True)

    plt.grid(True)

    return win_rates_figure, game_lengths_figure


def save_figures(win_rates_figure, game_lengths_figure):
    win_rates_figure.savefig(
        destination_path_for_average_win_rate_graph,
        bbox_inches='tight',
        pad_inches=0.05
    )

    game_lengths_figure.savefig(
        destination_path_for_average_game_length_graph,
        bbox_inches='tight',
        pad_inches=0.05
    )

    plt.show()


def determine_wilson_cis(k: int, n: int):
	result = st.binomtest(k, n=n, p=0.5)
	interval = result.proportion_ci(0.95, "wilson")
	return (np.round(interval[0], 3), np.round(interval[1], 3))


def determine_neyman_cis_for_game_lengths(absolute_history: list):
	game_lengths = [item["game_length"] for item in absolute_history]
	interval = st.t.interval(0.95, len(game_lengths)-1, loc=np.mean(game_lengths), scale=st.sem(game_lengths))
	return (np.round(interval[0], 3), np.round(interval[1], 3))


learning_rates = [0.003, 0.001, 0.0005, 0.0001]
final_win_rates_ppo = []
final_win_rates_ppo_cis = []
final_win_rates_random = []
final_win_rates_random_cis = []
final_game_lengths = []
final_game_lengths_cis = []

for learning_rate in learning_rates:
    print(learning_rate)
    path_to_average_history = "ppo_vs_random_alternating_player_order/" + str(learning_rate).replace(".", "_") + "/average_history.json"
    path_to_absolute_history = "ppo_vs_random_alternating_player_order/" + str(learning_rate).replace(".", "_") + "/absolute_history.json"
    average_history = []
    absolute_history = []

    with open(path_to_average_history) as f:
        average_history = json.load(f)["average_history"]

    with open(path_to_absolute_history) as f:
        absolute_history = json.load(f)["absolute_history"]

    last_average_history_item = average_history[len(average_history) - 1]
    final_win_rate_ppo = last_average_history_item["win_rate_ppo"]
    final_win_rate_random = last_average_history_item["win_rate_random"]
    final_win_rate_ppo_ci = determine_wilson_cis(int(final_win_rate_ppo * len(average_history)), len(average_history))
    final_win_rate_random_ci = determine_wilson_cis(int(final_win_rate_random * len(average_history)), len(average_history))
    final_game_length = last_average_history_item["average_game_length"]
    final_game_length_ci = determine_neyman_cis_for_game_lengths(absolute_history)
    
    final_win_rates_ppo.append(final_win_rate_ppo)
    final_win_rates_ppo_cis.append(final_win_rate_ppo_ci)
    final_win_rates_random.append(final_win_rate_random)
    final_win_rates_random_cis.append(final_win_rate_random_ci)
    final_game_lengths.append(final_game_length)
    final_game_lengths_cis.append(final_game_length_ci)

print(learning_rates)
print("Win rates PPO:", final_win_rates_ppo)
print("Win rates PPO (CI):", final_win_rates_ppo_cis)
print("Win rates random:", final_win_rates_random)
print("Win rates random (CI):", final_win_rates_random_cis)
print("Game lengths:", final_game_lengths)
print("Game lengths (CI):", final_game_lengths_cis)
print("HÄ")

win_rates_figure, game_lengths_figure = generate_figures(
    learning_rates,
    final_win_rates_ppo,
    final_win_rates_ppo_cis,
    final_win_rates_random,
    final_win_rates_random_cis,
    final_game_lengths,
    final_game_lengths_cis
)

save_figures(win_rates_figure, game_lengths_figure)

