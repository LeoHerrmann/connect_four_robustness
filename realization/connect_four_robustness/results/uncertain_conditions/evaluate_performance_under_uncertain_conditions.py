import json
import matplotlib
#matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
from scipy import stats as st
import numpy as np

def read_average_history(directory: str) -> list[dict]:
	file_path = directory + "average_history.json"

	with open(file_path) as f:
		average_history = json.load(f)["average_history"]

	return average_history


def determine_wilson_ci(k: int, n: int):
	result = st.binomtest(k, n=n, p=0.5)
	interval = result.proportion_ci(0.95, "wilson")
	return (np.round(interval[0], 3), np.round(interval[1], 3))


def win_rate_to_win_rate_loss(win_rate_with_uncertainty: float, win_rate_without_uncertainty: float) -> float:
	return (win_rate_without_uncertainty - win_rate_with_uncertainty) / (win_rate_without_uncertainty - 0.5)


def generate_and_save_figure(
		uncertainty_levels: list[int],
		uncertainty_variable_label: str,
		value_variable_label: str,
		final_win_rates_mcts: list[float],
		final_win_rates_mcts_cis: list[tuple],
		final_win_rates_ppo: list[float],
		final_win_rates_ppo_cis: list[tuple],
		destination_file_name: str
):
	matplotlib.rcParams["figure.dpi"] = 300
	matplotlib.rcParams["savefig.dpi"] = 300

	font = {'family': 'sans-serif',
			'weight': 'bold',
			'size': 16}

	matplotlib.rc('font', **font)

	final_win_rates_mcts_times_hundred = [win_rate * 100 for win_rate in final_win_rates_mcts]
	final_win_rates_mcts_cis_times_hundred = [(interval[0] * 100, interval[1] * 100) for interval in
											  final_win_rates_mcts_cis]
	final_win_rates_ppo_times_hundred = [win_rate * 100 for win_rate in final_win_rates_ppo]
	final_win_rates_ppo_cis_times_hundred = [(interval[0] * 100, interval[1] * 100) for interval in
												final_win_rates_ppo_cis]

	mcts_win_rates_lowers = [interval[0] for interval in final_win_rates_mcts_cis_times_hundred]
	mcts_win_rates_uppers = [interval[1] for interval in final_win_rates_mcts_cis_times_hundred]
	ppo_win_rates_lowers = [interval[0] for interval in final_win_rates_ppo_cis_times_hundred]
	ppo_win_rates_uppers = [interval[1] for interval in final_win_rates_ppo_cis_times_hundred]

	win_rates_figure = plt.figure(1)
	plt.plot(uncertainty_levels, final_win_rates_mcts_times_hundred, marker="o", label="MCTS", color="tab:blue",
			 markersize=3)
	plt.plot(uncertainty_levels, final_win_rates_ppo_times_hundred, marker="o", label="PPO", color="tab:orange",
			 markersize=3)

	for i in range(len(uncertainty_levels)):
		simulation_count = uncertainty_levels[i]
		plt.plot([simulation_count, simulation_count], [mcts_win_rates_lowers[i], mcts_win_rates_uppers[i]],
				 color="tab:blue", marker="_")
		plt.plot([simulation_count, simulation_count], [ppo_win_rates_lowers[i], ppo_win_rates_uppers[i]],
				 color="tab:orange", marker="_")

	plt.ylabel(value_variable_label)
	plt.xlabel(uncertainty_variable_label)
	plt.grid(True)
	plt.legend()

	win_rates_figure.savefig(
		destination_file_name,
		bbox_inches='tight',
		pad_inches=0
	)

	plt.show(block=False)
	plt.clf()
	plt.close()


# Evaluate performance for uncertain observations

read_directory = "uncertain_observations/alternating_player_order/"
uncertainty_levels = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

mcts_win_rates = []
mcts_win_rates_cis = []
mcts_win_rate_losses = []
mcts_win_rate_losses_cis = []
ppo_win_rates = []
ppo_win_rates_cis = []
ppo_win_rate_losses = []
ppo_win_rate_losses_cis = []

mcts_win_rate_without_uncertainty = read_average_history(read_directory + f"mcts_vs_random_0_0/")[-1]["win_rate_method"]
ppo_win_rate_without_uncertainty = read_average_history(read_directory + f"ppo_vs_random_0_0/")[-1]["win_rate_method"]

for uncertainty_level in uncertainty_levels:
	mcts_average_history = read_average_history(read_directory + f"mcts_vs_random_{uncertainty_level}_0/")
	mcts_win_rate = mcts_average_history[-1]["win_rate_method"]
	mcts_win_rate_ci = determine_wilson_ci(int(mcts_win_rate * len(mcts_average_history)), len(mcts_average_history))
	mcts_win_rate_loss = win_rate_to_win_rate_loss(mcts_win_rate, mcts_win_rate_without_uncertainty)
	mcts_win_rate_loss_ci = (
		win_rate_to_win_rate_loss(mcts_win_rate_ci[1], mcts_win_rate_without_uncertainty),
		win_rate_to_win_rate_loss(mcts_win_rate_ci[0], mcts_win_rate_without_uncertainty)
	)

	ppo_average_history = read_average_history(read_directory + f"ppo_vs_random_{uncertainty_level}_0/")
	ppo_win_rate = ppo_average_history[-1]["win_rate_method"]
	ppo_win_rate_ci = determine_wilson_ci(int(ppo_win_rate * len(ppo_average_history)), len(ppo_average_history))
	ppo_win_rate_loss = win_rate_to_win_rate_loss(ppo_win_rate, ppo_win_rate_without_uncertainty)
	ppo_win_rate_loss_ci = (
		win_rate_to_win_rate_loss(ppo_win_rate_ci[1], ppo_win_rate_without_uncertainty),
		win_rate_to_win_rate_loss(ppo_win_rate_ci[0], ppo_win_rate_without_uncertainty)
	)

	mcts_win_rates.append(mcts_win_rate)
	mcts_win_rates_cis.append(mcts_win_rate_ci)
	mcts_win_rate_losses.append(mcts_win_rate_loss)
	mcts_win_rate_losses_cis.append(mcts_win_rate_loss_ci)
	ppo_win_rates.append(ppo_win_rate)
	ppo_win_rates_cis.append(ppo_win_rate_ci)
	ppo_win_rate_losses.append(ppo_win_rate_loss)
	ppo_win_rate_losses_cis.append(ppo_win_rate_loss_ci)

generate_and_save_figure(
	uncertainty_levels,
	"Anz. fehlerhafter Spielsteinplatzierungen",
	"Gewinnrate [%]",
	mcts_win_rates,
	mcts_win_rates_cis,
	ppo_win_rates,
	ppo_win_rates_cis,
	"uncertain_observations/win_rates.png"
)

generate_and_save_figure(
	uncertainty_levels,
	"Anz. fehlerhafter Spielsteinplatzierungen",
"Gewinnratenverlust [%]",
	mcts_win_rate_losses,
	mcts_win_rate_losses_cis,
	ppo_win_rate_losses,
	ppo_win_rate_losses_cis,
	"uncertain_observations/win_rate_losses.png"
)

# Evaluate performance for uncertain actions

read_directory = "uncertain_actions/alternating_player_order/"
uncertainty_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

mcts_win_rates = []
mcts_win_rates_cis = []
mcts_win_rate_losses = []
mcts_win_rate_losses_cis = []
ppo_win_rates = []
ppo_win_rates_cis = []
ppo_win_rate_losses = []
ppo_win_rate_losses_cis = []

mcts_win_rate_without_uncertainty = read_average_history(read_directory + f"mcts_vs_random_0_0/")[-1]["win_rate_method"]
ppo_win_rate_without_uncertainty = read_average_history(read_directory + f"ppo_vs_random_0_0/")[-1]["win_rate_method"]

for uncertainty_level in uncertainty_levels:
	mcts_average_history = read_average_history(read_directory + f"mcts_vs_random_0_{uncertainty_level}/")
	mcts_win_rate = mcts_average_history[-1]["win_rate_method"]
	mcts_win_rate_ci = determine_wilson_ci(int(mcts_win_rate * len(mcts_average_history)), len(mcts_average_history))
	mcts_win_rate_loss = win_rate_to_win_rate_loss(mcts_win_rate, mcts_win_rate_without_uncertainty)
	mcts_win_rate_loss_ci = (
		win_rate_to_win_rate_loss(mcts_win_rate_ci[1], mcts_win_rate_without_uncertainty),
		win_rate_to_win_rate_loss(mcts_win_rate_ci[0], mcts_win_rate_without_uncertainty)
	)

	ppo_average_history = read_average_history(read_directory + f"ppo_vs_random_0_{uncertainty_level}/")
	ppo_win_rate = ppo_average_history[-1]["win_rate_method"]
	ppo_win_rate_ci = determine_wilson_ci(int(ppo_win_rate * len(ppo_average_history)), len(ppo_average_history))
	ppo_win_rate_loss = win_rate_to_win_rate_loss(ppo_win_rate, ppo_win_rate_without_uncertainty)
	ppo_win_rate_loss_ci = (
		win_rate_to_win_rate_loss(ppo_win_rate_ci[1], ppo_win_rate_without_uncertainty),
		win_rate_to_win_rate_loss(ppo_win_rate_ci[0], ppo_win_rate_without_uncertainty)
	)

	mcts_win_rates.append(mcts_win_rate)
	mcts_win_rates_cis.append(mcts_win_rate_ci)
	mcts_win_rate_losses.append(mcts_win_rate_loss)
	mcts_win_rate_losses_cis.append(mcts_win_rate_loss_ci)
	ppo_win_rates.append(ppo_win_rate)
	ppo_win_rates_cis.append(ppo_win_rate_ci)
	ppo_win_rate_losses.append(ppo_win_rate_loss)
	ppo_win_rate_losses_cis.append(ppo_win_rate_loss_ci)

generate_and_save_figure(
	uncertainty_levels,
	"Wahrsch. für zufällige Aktion [%]",
	"Gewinnrate [%]",
	mcts_win_rates,
	mcts_win_rates_cis,
	ppo_win_rates,
	ppo_win_rates_cis,
	"uncertain_actions/win_rates.png"
)

generate_and_save_figure(
	uncertainty_levels,
	"Wahrsch. für zufällige Aktion [%]",
"Gewinnratenverlust [%]",
	mcts_win_rate_losses,
	mcts_win_rate_losses_cis,
	ppo_win_rate_losses,
	ppo_win_rate_losses_cis,
	"uncertain_actions/win_rate_losses.png"
)
