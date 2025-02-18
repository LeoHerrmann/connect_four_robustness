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

                        if not current_field_is_occupied and not field_below_is_occupied and number_of_free_fields_in_column >= 2:
                            coordinates_of_addable_pieces.append((row_index, column_index))

            rn = random.randint(0, len(coordinates_of_removable_pieces) + len(coordinates_of_addable_pieces) - 1)

            if rn < len(coordinates_of_removable_pieces):
                row_index_of_piece_to_remove = coordinates_of_removable_pieces[rn][0]
                column_index_of_piece_to_remove = coordinates_of_removable_pieces[rn][1]
                distorted_state[row_index_of_piece_to_remove][column_index_of_piece_to_remove] = [0, 0]
            else:
                row_index_of_piece_to_add = coordinates_of_addable_pieces[rn - len(coordinates_of_removable_pieces)][0]
                column_index_of_piece_to_add = coordinates_of_addable_pieces[rn - len(coordinates_of_removable_pieces)][1]

                new_piece = [0, 1]

                if random.randint(0, 1) == 0:
                    new_piece = [1, 0]

                distorted_state[row_index_of_piece_to_add][column_index_of_piece_to_add] = new_piece

        return distorted_state


def free_fields_in_column(state, column_index: int) -> int:
    free_fields_count = 0

    for row_index in range(len(state)):
        field_in_column = state[row_index][column_index]
        field_in_column_is_occupied = field_in_column[0] == 1 or field_in_column[1] == 1

        if not field_in_column_is_occupied:
            free_fields_count += 1

    return free_fields_count
