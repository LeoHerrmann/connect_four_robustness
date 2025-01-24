from pettingzoo.classic import connect_four_v3
import matplotlib.pyplot as plt


def play_game(env):
        env.reset()
        game_length = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            game_length += 1

            if termination or truncation:
                statistics = {
                    "result": "",
                    "game_length": game_length
                }

                if env.rewards[env.possible_agents[0]] != env.rewards[env.possible_agents[1]]:
                    winner = max(env.rewards, key=env.rewards.get)
                    print("Der Gewinner ist: " + winner)

                    if winner == env.agents[0]:
                        statistics["result"] = "player_0"
                    elif winner == env.agents[1]:
                        statistics["result"] = "player_1"

                else:
                    print("Das Ergebnis ist unentschieden")
                    statistics["result"] = "draw"

                return statistics

            else:
                mask = observation["action_mask"]

                #print(agent + ": Ich sehe folgende Beobachtung: ")
                #print(observation)

                if agent == env.possible_agents[0]:
                    action = env.action_space(agent).sample(mask)
                else:
                    #print("Folgende Aktionen sind möglich: ")
                    #print(env.action_space(agent))
                    #action = int(input("Welche Aktion möchtest du ausführen? "))
                    action = env.action_space(agent).sample(mask)

                # print(agent + ": Ich habe folgende Aktion ausgewählt: " + str(action))

            env.step(action)


def play_games(number_of_games):
    history = []
    player_0_win_count = 0
    player_1_win_count = 0
    draw_count = 0
    average_game_length = 0

    env = connect_four_v3.env()#render_mode="human")

    for i in range(number_of_games):
        game_statistics = play_game(env)

        if game_statistics["result"] == "player_0":
            player_0_win_count += 1
        elif game_statistics["result"] == "player_1":
            player_1_win_count += 1
        elif game_statistics["result"] == "draw":
            draw_count += 1

        player_0_win_rate = player_0_win_count / (len(history) + 1)
        player_1_win_rate = player_1_win_count / (len(history) + 1)
        average_game_length = (game_statistics["game_length"] - average_game_length) / (len(history) + 1) + average_game_length

        history.append({
            "player_0_win_rate": player_0_win_rate,
            "player_1_win_rate": player_1_win_rate,
            "average_game_length": average_game_length
        })

        print(game_statistics)
        print(history)

    generate_graph(history)

    env.close()


def generate_graph(history):
    player_0_win_rates = [item['player_0_win_rate'] for item in history]
    player_1_win_rates = [item['player_1_win_rate'] for item in history]
    average_game_lengths = [item['average_game_length'] for item in history]
    game_indices = range(1, len(history) + 1)

    plt.figure(1)
    plt.plot(game_indices, player_0_win_rates, label='Player 0')
    plt.plot(game_indices, player_1_win_rates, label='Player 1')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.legend()

    plt.figure(2)
    plt.plot(game_indices, average_game_lengths, color='black', label='Average Game Length')
    plt.ylabel('Average Game Length')
    plt.xlabel('Game Index')
    plt.grid(True)

    plt.grid(True)
    plt.show()

    # fig, axs = plt.subplots(2, sharex=True)

    # axs[0].plot(game_indices, player_0_win_rates, label='Player 0')
    # axs[0].plot(game_indices, player_1_win_rates, label='Player 1')
    # axs[0].set_ylabel('Win Rate')
    # axs[0].grid(True)
    # axs[0].legend()

    # axs[1].plot(game_indices, average_game_lengths, color='black', label='Average Game Length')
    # axs[1].set_ylabel('Average Game Length')
    # axs[1].set_xlabel('Game Index')
    # axs[1].grid(True)

    # plt.grid(True)
    # plt.show()


play_games(1000)

# Was brauche ich jetzt?
# - Versionskontrolle
# - Abwechselnde Anfangsspieler, am besten konfigurierbar