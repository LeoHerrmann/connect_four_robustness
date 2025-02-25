import json
import glob
import pathlib
import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt


def read_absolute_history(directory: str) -> list[dict]:
	file_pattern = directory + "*absolute_history.json"
	matching_files = glob.glob(file_pattern)

	if len(matching_files) != 1:
		raise FileNotFoundError(f"Need exactly 1 file matching {file_pattern} but found {len(matching_files)}")

	with open(matching_files[0]) as f:
		absolute_history = json.load(f)["absolute_history"]

	return absolute_history


def merge_absolute_history(m_vs_r_abs_his: list[dict], r_vs_m_abs_his: list[dict]) -> list[dict]:
	merged_absolute_history = []

	for i in range(len(m_vs_r_abs_his)):
		winner_history_item_1 = m_vs_r_abs_his[i]["winner"]

		if winner_history_item_1 == "player_0":
			winner_new_history_item_1 = "method"
		elif winner_history_item_1 == "player_1":
			winner_new_history_item_1 = "random"
		elif winner_history_item_1 == "draw":
			winner_new_history_item_1 = "draw"
		else:
			raise ValueError(f"Winner must be either player_0, player_1, draw. But it was {winner_history_item_1}")

		winner_history_item_2 = r_vs_m_abs_his[i]["winner"]

		if winner_history_item_2 == "player_1":
			winner_new_history_item_2 = "method"
		elif winner_history_item_2 == "player_0":
			winner_new_history_item_2 = "random"
		elif winner_history_item_2 == "draw":
			winner_new_history_item_2 = "draw"
		else:
			raise ValueError(f"Winner must be either player_0, player_1, draw. But it was {winner_history_item_2}")

		new_history_item_1 = {
			"winner": winner_new_history_item_1,
			"game_length": m_vs_r_abs_his[i]["game_length"]
		}

		new_history_item_2 = {
			"winner": winner_new_history_item_2,
			"game_length": r_vs_m_abs_his[i]["game_length"]
		}

		merged_absolute_history.append(new_history_item_1)
		merged_absolute_history.append(new_history_item_2)

	return merged_absolute_history


def generate_avgerage_history(abs_his: list[dict]) -> list[dict]:
	average_history = []

	total_wins_method = 0
	total_wins_random = 0
	total_game_length = 0

	for i in range(len(abs_his)):
		game_count = i + 1

		total_game_length += abs_his[i]["game_length"]

		if abs_his[i]["winner"] == "method":
			total_wins_method += 1
		elif abs_his[i]["winner"] == "random":
			total_wins_random += 1

		new_history_item = {
			"win_rate_method": total_wins_method / game_count,
			"win_rate_random": total_wins_random / game_count,
			"average_game_length": total_game_length / game_count
		}

		average_history.append(new_history_item)

	return average_history


def generate_and_save_avgerage_history_figures(average_history: list[dict], method_name: str, destination_directory: str) -> None:
	matplotlib.rcParams["figure.dpi"] = 300
	matplotlib.rcParams["savefig.dpi"] = 300

	font = {'family': 'sans-serif',
			'weight': 'bold',
			'size': 16}

	matplotlib.rc('font', **font)

	player_0_win_rates = [(item["win_rate_method"] * 100) for item in average_history]
	player_1_win_rates = [(item["win_rate_random"] * 100) for item in average_history]
	average_game_lengths = [item["average_game_length"] for item in average_history]
	game_indices = range(1, len(average_history) + 1)

	win_rates_figure = plt.figure(1)
	plt.plot(game_indices, player_0_win_rates, label=method_name.upper())
	plt.plot(game_indices, player_1_win_rates, label="Zufällig")
	plt.ylabel("Gewinnrate [%]")
	plt.xlabel("Anzahl der Spiele")
	plt.grid(True)
	plt.legend()

	win_rates_figure.savefig(
		destination_directory + "win_rates_figure.jpg",
		bbox_inches='tight',
		pad_inches=0
	)

	plt.show(block=False)
	plt.clf()
	plt.close()

	game_length_figure = plt.figure(2)
	plt.plot(game_indices, average_game_lengths, color="black", label="Average Game Length")
	plt.ylabel("Durchschnittliche Spieldauer")
	plt.xlabel("Anzahl der Spiele")
	plt.grid(True)

	game_length_figure.savefig(
		destination_directory + "game_length_figure.jpg",
		bbox_inches='tight',
		pad_inches=0
	)

	plt.show(block=False)
	plt.clf()
	plt.close()


def save_history(history: list[dict], destination_path: str, key: str) -> None:
	pathlib.Path( "/".join(destination_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)

	with open(destination_path, "w+") as f:
		json.dump({key: history}, f)


methods = ["mcts", "ppo"]

# Merge histories with uncertain actions

uncertainty = "actions"
read_directory = f"uncertain_{uncertainty}/constant_player_order/"

for method in methods:
	method_read_directory = read_directory + f"{method}_vs_random_uncertain_{uncertainty}/"

	for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
		m_vs_r_abs_his = read_absolute_history(method_read_directory + f"{method}_vs_random_0_{i}/")
		r_vs_m_abs_his = read_absolute_history(method_read_directory + f"random_vs_{method}_0_{i}/")
		write_dir = f"uncertain_{uncertainty}/alternating_player_order/{method}_vs_random_0_{i}/"
		
		merged_abs_his = merge_absolute_history(m_vs_r_abs_his, r_vs_m_abs_his)
		merged_avg_his = generate_avgerage_history(merged_abs_his)
		generate_and_save_avgerage_history_figures(merged_avg_his, method, write_dir)

		save_history(merged_abs_his, write_dir + "absolute_history.json", "absolute_history")
		save_history(merged_avg_his, write_dir + "average_history.json", "average_history")

# Merge histories with uncertain observations

uncertainty = "observations"
read_directory = f"uncertain_{uncertainty}/constant_player_order/"

for method in methods:
	method_read_directory = read_directory + f"{method}_vs_random_uncertain_{uncertainty}/"

	for i in [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
		m_vs_r_abs_his = read_absolute_history(method_read_directory + f"{method}_vs_random_{i}_0/")
		r_vs_m_abs_his = read_absolute_history(method_read_directory + f"random_vs_{method}_{i}_0/")
		write_dir = f"uncertain_{uncertainty}/alternating_player_order/{method}_vs_random_{i}_0/"
		
		merged_abs_his = merge_absolute_history(m_vs_r_abs_his, r_vs_m_abs_his)
		merged_avg_his = generate_avgerage_history(merged_abs_his)
		generate_and_save_avgerage_history_figures(merged_avg_his, method, write_dir)
		
		save_history(merged_abs_his, write_dir + "absolute_history.json", "absolute_history")
		save_history(merged_avg_his, write_dir + "average_history.json", "average_history")
