import random
import copy
import numpy


class DistortionGenerator:
    def __init__(self, number_of_fields_to_distort: int, probability_of_distorting_actions: float):
        if number_of_fields_to_distort < 0:
            raise ValueError("Number of fields to distort must be at least zero")

        if probability_of_distorting_actions > 1 or probability_of_distorting_actions < 0:
            raise ValueError("Probability of distorting actions must be between zero and one")

        self.number_of_fields_to_distort = number_of_fields_to_distort
        self.probability_of_distorting_actions = probability_of_distorting_actions

    def distort_action(self, action: int, action_mask: list[int]) -> int:
        """Returns an action distorted with the respective distortion probability"""

        if random.randint(0, 99) / 100 >= self.probability_of_distorting_actions:
            return action

        legal_actions = [i for i, is_legal in enumerate(action_mask) if is_legal]
        chosen_action = random.choice(legal_actions)
        return chosen_action

    def distort_state(self, state: numpy.ndarray) -> numpy.ndarray:
        """Returns a deep copy of the given state which is distorted with the respective distortion probability"""

        distorted_state = copy.deepcopy(state)

        for i in range(self.number_of_fields_to_distort):
            coordinates_of_removable_pieces = []
            coordinates_of_addable_pieces = []

            for row_index in range(len(distorted_state)):
                for column_index in range(len(distorted_state[row_index])):
                    current_field = distorted_state[row_index][column_index]
                    current_field_is_occupied = current_field[0] == 1 or current_field[1] == 1
                    number_of_free_fields_in_column = free_fields_in_column(distorted_state, column_index)

                    if row_index > 0:
                        field_above = distorted_state[row_index - 1][column_index]
                        field_above_is_occupied = field_above[0] == 1 or field_above[1] == 1

                        if current_field_is_occupied and field_above_is_occupied and number_of_free_fields_in_column != 0:
                            coordinates_of_removable_pieces.append((row_index, column_index))

                    if row_index < len(distorted_state) - 1:
                        field_below = distorted_state[row_index + 1][column_index]
                        field_below_is_occupied = field_below[0] == 1 or field_below[1] == 1
                        player_for_which_current_field_builds_diagonal_chain = determine_player_for_which_field_builds_diagonal_chain(distorted_state, row_index, column_index)

                        if not current_field_is_occupied and not field_below_is_occupied and number_of_free_fields_in_column >= 2:
                            if player_for_which_current_field_builds_diagonal_chain == -1:
                                coordinates_of_addable_pieces.append((row_index, column_index, 0))
                                coordinates_of_addable_pieces.append((row_index, column_index, 1))
                            elif player_for_which_current_field_builds_diagonal_chain == 0:
                                coordinates_of_addable_pieces.append((row_index, column_index, 1))
                            elif player_for_which_current_field_builds_diagonal_chain == 1:
                                coordinates_of_addable_pieces.append((row_index, column_index, 0))


            rn = random.randint(0, len(coordinates_of_removable_pieces) + len(coordinates_of_addable_pieces) - 1)

            if rn < len(coordinates_of_removable_pieces):
                row_index_of_piece_to_remove = coordinates_of_removable_pieces[rn][0]
                column_index_of_piece_to_remove = coordinates_of_removable_pieces[rn][1]
                distorted_state[row_index_of_piece_to_remove][column_index_of_piece_to_remove] = [0, 0]
            else:
                row_index_of_piece_to_add = coordinates_of_addable_pieces[rn - len(coordinates_of_removable_pieces)][0]
                column_index_of_piece_to_add = coordinates_of_addable_pieces[rn - len(coordinates_of_removable_pieces)][1]
                player_index_of_piece_to_add = coordinates_of_addable_pieces[rn - len(coordinates_of_removable_pieces)][2]

                if player_index_of_piece_to_add == 0:
                    new_piece = [1, 0]
                else:
                    new_piece = [0, 1]

                distorted_state[row_index_of_piece_to_add][column_index_of_piece_to_add] = new_piece

        return distorted_state


def determine_player_for_which_field_builds_diagonal_chain(distorted_state, target_field_row_index: int, target_field_column_index: int) -> int:
    """Checks whether placing the target field results in building a diagonal chain.
    Returns -1 when no diagonal chain is formed.
    Returns 0 when diagonal chain for player 0 is formed.
    Returns 1 when diagonal chain for player 1 is formed."""

    for row_index in range(len(distorted_state) - 3):
        for column_index in range(len(distorted_state[row_index]) - 3):
            neg_sloped_chain_coords = [
                (row_index, column_index),
                (row_index + 1, column_index + 1),
                (row_index + 2, column_index + 2),
                (row_index + 3, column_index + 3),
            ]

            coordinates_contain_target_field = False

            for coordinates in neg_sloped_chain_coords:
                if coordinates[0] == target_field_row_index and coordinates[1] == target_field_column_index:
                    coordinates_contain_target_field = True

            if coordinates_contain_target_field:
                # Check whether the three coordinates which are not the target coordinates have the same value

                target_field_builds_diagonal_chain_for_player_0 = True

                for coordinates in neg_sloped_chain_coords:
                    coordinates_are_occupied_by_player_0 = distorted_state[coordinates[0]][coordinates[1]][0] == 1
                    coordinates_are_target_coordinates = coordinates[0] == target_field_row_index and coordinates[1] == target_field_column_index

                    if coordinates_are_occupied_by_player_0 or coordinates_are_target_coordinates:
                        target_field_builds_diagonal_chain_for_player_0 = True
                    else:
                        target_field_builds_diagonal_chain_for_player_0 = False
                        break

                if target_field_builds_diagonal_chain_for_player_0:
                    return 0

                target_field_builds_diagonal_chain_for_player_1 = True

                for coordinates in neg_sloped_chain_coords:
                    coordinates_are_occupied_by_player_1 = distorted_state[coordinates[0]][coordinates[1]][1] == 1
                    coordinates_are_target_coordinates = coordinates[0] == target_field_row_index and coordinates[1] == target_field_column_index

                    if coordinates_are_occupied_by_player_1 or coordinates_are_target_coordinates:
                        target_field_builds_diagonal_chain_for_player_1 = True
                    else:
                        target_field_builds_diagonal_chain_for_player_1 = False
                        break

                if target_field_builds_diagonal_chain_for_player_1:
                    return 1

    for row_index in range(3, len(distorted_state)):
        for column_index in range(len(distorted_state[row_index]) - 3):
            pos_sloped_chain_coords = [
                (row_index, column_index),
                (row_index - 1, column_index + 1),
                (row_index - 2, column_index + 2),
                (row_index - 3, column_index + 3),
            ]

            coordinates_contain_target_field = False

            for coordinates in pos_sloped_chain_coords:
                if coordinates[0] == target_field_row_index and coordinates[1] == target_field_column_index:
                    coordinates_contain_target_field = True

            if coordinates_contain_target_field:
                # Check whether the three coordinates which are not the target coordinates have the same value

                target_field_builds_diagonal_chain_for_player_0 = True

                for coordinates in pos_sloped_chain_coords:
                    coordinates_are_occupied_by_player_0 = distorted_state[coordinates[0]][coordinates[1]][0] == 1
                    coordinates_are_target_coordinates = coordinates[0] == target_field_row_index and coordinates[1] == target_field_column_index

                    if coordinates_are_occupied_by_player_0 or coordinates_are_target_coordinates:
                        target_field_builds_diagonal_chain_for_player_0 = True
                    else:
                        target_field_builds_diagonal_chain_for_player_0 = False
                        break

                if target_field_builds_diagonal_chain_for_player_0:
                    return 0

                target_field_builds_diagonal_chain_for_player_1 = True

                for coordinates in pos_sloped_chain_coords:
                    coordinates_are_occupied_by_player_1 = distorted_state[coordinates[0]][coordinates[1]][1] == 1
                    coordinates_are_target_coordinates = coordinates[0] == target_field_row_index and coordinates[1] == target_field_column_index

                    if coordinates_are_occupied_by_player_1 or coordinates_are_target_coordinates:
                        target_field_builds_diagonal_chain_for_player_1 = True
                    else:
                        target_field_builds_diagonal_chain_for_player_1 = False
                        break

                if target_field_builds_diagonal_chain_for_player_1:
                    return 1

    return -1


def free_fields_in_column(state, column_index: int) -> int:
    free_fields_count = 0

    for row_index in range(len(state)):
        field_in_column = state[row_index][column_index]
        field_in_column_is_occupied = field_in_column[0] == 1 or field_in_column[1] == 1

        if not field_in_column_is_occupied:
            free_fields_count += 1

    return free_fields_count
