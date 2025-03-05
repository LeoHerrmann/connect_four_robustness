import json

read_file_path = "random_vs_random_1/constant_player_order/20250223-223154_absolute_history.json"
write_file_path = "random_vs_random_1/alternating_player_order/20250223-223154_absolute_history.json"

absolute_history_with_constant_player_order = []
absolute_history_with_alternating_player_order = []

with open(read_file_path) as read_file:
	d = json.load(read_file)
	absolute_history_with_constant_player_order = d["absolute_history"]

for i in range(len(absolute_history_with_constant_player_order)):
	if i % 2 == 0:
		absolute_history_with_alternating_player_order.append(
			absolute_history_with_constant_player_order[i]
		)
	else:
		original_winner = absolute_history_with_constant_player_order[i]["winner"]
		
		if original_winner == "player_0":
			alternated_winner = "player_1"
		elif original_winner == "player_1":
			alternated_winner = "player_0"
		elif original_winner == "draw":
			alternated_winner = "draw"
		else:
			print("Error. Original winner was neither player_0 nor player_1")
			print("Original winner was", original_winner)
			print("Aborting")
			exit()

		alternated_history_item = {
			"winner": alternated_winner,
			"game_length": absolute_history_with_constant_player_order[i]["game_length"]
		}
		
		absolute_history_with_alternating_player_order.append(
			alternated_history_item
		)

with open(write_file_path, "w+") as write_file:
	json.dump(
		{
			"absolute_history": absolute_history_with_alternating_player_order
		},
		write_file
	)

