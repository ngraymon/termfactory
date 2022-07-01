
def compute_m0_n1_fully_connected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the fully_connected terms."""

    fully_connected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return fully_connected_opt_path_list


def compute_m0_n1_linked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the linked_disconnected terms."""

    linked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return linked_disconnected_opt_path_list


def compute_m0_n1_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the unlinked_disconnected terms."""

    unlinked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        if truncation.singles:
            unlinked_disconnected_opt_path_list.append(oe.contract_expression((A, A), (A, A, N)))
    else:
        raise NotImplementedError('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return unlinked_disconnected_opt_path_list

