import json
import matplotlib
matplotlib.use('gtk3agg') 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

destination_path_for_average_win_rate_graph = "mcts_vs_mcts/win_rate_vs_n_simulations.png"
destination_path_for_average_game_length_graph = "mcts_vs_mcts/game_length_vs_n_simulations.png"

def generate_figures(
    simulation_counts,
    final_win_rates_player_0,
    final_win_rates_player_0_cis,
    final_win_rates_player_1,
    final_win_rates_player_1_cis,
    final_game_lengths,
    final_game_lengths_cis
):
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = 300

    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 16}

    matplotlib.rc('font', **font)

    final_win_rates_player_0_times_hundred = [win_rate * 100 for win_rate in final_win_rates_player_0]
    final_win_rates_player_0_cis_times_hundred = [(interval[0] * 100, interval[1] * 100) for interval in final_win_rates_player_0_cis]
    final_win_rates_player_1_times_hundred = [win_rate * 100 for win_rate in final_win_rates_player_1]
    final_win_rates_player_1_cis_times_hundred = [(interval[0] * 100, interval[1] * 100) for interval in final_win_rates_player_1_cis]

    player_0_win_rates_lowers = [interval[0] for interval in final_win_rates_player_0_cis_times_hundred]
    player_0_win_rates_uppers = [interval[1] for interval in final_win_rates_player_0_cis_times_hundred]
    player_1_win_rates_lowers = [interval[0] for interval in final_win_rates_player_1_cis_times_hundred]
    player_1_win_rates_uppers = [interval[1] for interval in final_win_rates_player_1_cis_times_hundred]

    win_rates_figure = plt.figure(1)
    plt.ylim(0, 100)
    plt.plot(simulation_counts, final_win_rates_player_0_times_hundred, marker="o", label="Spieler 0", color="tab:blue", markersize=3)
    plt.plot(simulation_counts, final_win_rates_player_1_times_hundred, marker="o", label="Spieler 1", color="tab:orange", markersize=3)
    
    for i in range(len(simulation_counts)):
        simulation_count = simulation_counts[i]
        plt.plot([simulation_count, simulation_count], [player_0_win_rates_lowers[i], player_0_win_rates_uppers[i]], color="tab:blue", marker="_")
        plt.plot([simulation_count, simulation_count], [player_1_win_rates_lowers[i], player_1_win_rates_uppers[i]], color="tab:orange", marker="_")
    
    plt.ylabel("Gewinnrate / %")
    plt.xlabel("Anzahl der Simulationen pro Entscheidung")
    plt.grid(True)
    plt.legend()

    game_lengths_figure = plt.figure(2)
    plt.plot(simulation_counts, final_game_lengths, marker="o", label="Durchschnittliche Spieldauer", markersize=3)
    
    for i in range(len(simulation_counts)):
        simulation_count = simulation_counts[i]
        plt.plot([simulation_count, simulation_count], [final_game_lengths_cis[i][0], final_game_lengths_cis[i][1]], marker="_", color="tab:blue")
    
    plt.ylabel("Durchschn. Spieldauer / Züge")
    plt.xlabel("Anzahl der Simulationen pro Entscheidung")
    plt.grid(True)

    plt.grid(True)

    return win_rates_figure, game_lengths_figure


def save_figures(win_rates_figure, game_lengths_figure):
    win_rates_figure.savefig(
        destination_path_for_average_win_rate_graph,
        bbox_inches='tight',
        pad_inches=0
    )

    game_lengths_figure.savefig(
        destination_path_for_average_game_length_graph,
        bbox_inches='tight',
        pad_inches=0
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


simulation_counts = [50, 100, 250, 500, 750, 1000, 2500, 5000]
final_win_rates_player_0 = []
final_win_rates_player_0_cis = []
final_win_rates_player_1 = []
final_win_rates_player_1_cis = []
final_game_lengths = []
final_game_lengths_cis = []

for simulation_count in simulation_counts:
    print(simulation_count)
    path_to_average_history = "mcts_vs_mcts/mcts_vs_mcts_" + str(simulation_count) + "/average_history.json"
    path_to_absolute_history = "mcts_vs_mcts/mcts_vs_mcts_" + str(simulation_count) + "/absolute_history.json"

    average_history = []
    absolute_history = []

    with open(path_to_average_history) as f:
        average_history = json.load(f)["average_history"]

    with open(path_to_absolute_history) as f:
        absolute_history = json.load(f)["absolute_history"]

    last_average_history_item = average_history[len(average_history) - 1]
    final_win_rate_player_0 = last_average_history_item["player_0_win_rate"]
    final_win_rate_player_1 = last_average_history_item["player_1_win_rate"]
    final_win_rate_player_0_ci = determine_wilson_cis(int(final_win_rate_player_0 * len(average_history)), len(average_history))
    final_win_rate_player_1_ci = determine_wilson_cis(int(final_win_rate_player_1 * len(average_history)), len(average_history))
    final_game_length = last_average_history_item["average_game_length"]
    final_game_length_ci = determine_neyman_cis_for_game_lengths(absolute_history)
    
    final_win_rates_player_0.append(final_win_rate_player_0)
    final_win_rates_player_0_cis.append(final_win_rate_player_0_ci)
    final_win_rates_player_1.append(final_win_rate_player_1)
    final_win_rates_player_1_cis.append(final_win_rate_player_1_ci)
    final_game_lengths.append(final_game_length)
    final_game_lengths_cis.append(final_game_length_ci)

print(simulation_counts)
print("Win rates player 0:", final_win_rates_player_0)
print("Win rates player 0 (CI):", final_win_rates_player_0_cis)
print("Win rates player 1:", final_win_rates_player_1)
print("Win rates player 1 (CI):", final_win_rates_player_1_cis)
print("Game lengths:", final_game_lengths)
print("Game lengths (CI):", final_game_lengths_cis)

win_rates_figure, game_lengths_figure = generate_figures(
    simulation_counts,
    final_win_rates_player_0,
    final_win_rates_player_0_cis,
    final_win_rates_player_1,
    final_win_rates_player_1_cis,
    final_game_lengths,
    final_game_lengths_cis
)

save_figures(win_rates_figure, game_lengths_figure)

