import random
import copy


def distort_action(action: int, action_mask: list[int], probability: float):
    if probability > 1 or probability < 0:
        raise ValueError("probability must be between zero and one")

    if random.randint(0, 99) / 100 >= probability:
        print("DID NOT DISTORT ACTION")
        return action

    print("DID DISTORT ACTION")

    legal_actions = [i for i, is_legal in enumerate(action_mask) if is_legal]
    action = random.choice(legal_actions)
    return action


def distort_state(state, probability: float):
    """Returns a deep copy of the given state which is distorted with the given probability"""

    if probability > 1 or probability < 0:
        raise ValueError("probability must be between zero and one")

    if random.randint(0, 99) / 100 >= probability:
        print("DID NOT DISTORT STATE")
        return copy.deepcopy(state)

    print("DID DISTORT STATE")

    distorted_state = copy.deepcopy(state)
    coordinates_of_removable_pieces = []
    coordinates_of_addable_pieces = []

    for row_index in range(len(distorted_state)):
        for column_index in range(len(distorted_state[row_index])):
            current_field = distorted_state[row_index][column_index]
            current_field_is_occupied = current_field[0] == 1 or current_field[1] == 1

            if row_index > 0:
                field_above = distorted_state[row_index - 1][column_index]
                field_above_is_occupied = field_above[0] == 1 or field_above[1] == 1
                if current_field_is_occupied and field_above_is_occupied:
                    coordinates_of_removable_pieces.append((row_index, column_index))

            if row_index < len(distorted_state) - 1:
                field_below = distorted_state[row_index + 1][column_index]
                field_below_is_occupied = field_below[0] == 1 or field_below[1] == 1

                if not current_field_is_occupied and not field_below_is_occupied:
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
