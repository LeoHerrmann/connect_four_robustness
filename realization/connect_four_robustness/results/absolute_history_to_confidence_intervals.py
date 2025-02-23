import json
import numpy as np, scipy.stats as st

read_file_path = "random_vs_random_0/constant_player_order/20250223-202221_absolute_history.json"

absolute_history = []

with open(read_file_path) as read_file:
	d = json.load(read_file)
	absolute_history = d["absolute_history"]

game_lengths = [item["game_length"] for item in absolute_history]
average_game_length = sum(game_lengths) / len(game_lengths)

confidence_interval = st.t.interval(
	0.95,
	len(game_lengths) - 1,
	loc=np.mean(game_lengths),
	scale=st.sem(game_lengths)
)

print("Average game length: ", average_game_length)
print("Confidence interval:", confidence_interval[0], "-", confidence_interval[1])

