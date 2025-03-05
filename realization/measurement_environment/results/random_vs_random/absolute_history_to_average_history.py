import json

read_file_path = "random_vs_random_1/alternating_player_order/20250223-223154_absolute_history.json"
write_file_path = "random_vs_random_1/alternating_player_order/20250223-223154_average_history.json"

absolute_history = []
average_history = []

with open(read_file_path) as read_file:
	d = json.load(read_file)
	absolute_history = d["absolute_history"]

for i in range(len(absolute_history)):
	absolute_history_until_i = absolute_history[:i+1]

	player_0_wins = 0
	player_1_wins = 0
	accumulated_game_length = 0
	
	for item in absolute_history_until_i:
		if item["winner"] == "player_0":
			player_0_wins += 1
		elif item["winner"] == "player_1":
			player_1_wins += 1
		
		accumulated_game_length += item["game_length"]
	
	player_0_win_rate = player_0_wins / len(absolute_history_until_i)
	player_1_win_rate = player_1_wins / len(absolute_history_until_i)
	average_game_length = accumulated_game_length / len(absolute_history_until_i)
	
	average_history.append({
		"player_0_win_rate": player_0_win_rate,
		"player_1_win_rate": player_1_win_rate,
		"average_game_length": average_game_length
	})

with open(write_file_path, "w+") as write_file:
	json.dump(
		{
			"average_history": average_history
		},
		write_file
	)


