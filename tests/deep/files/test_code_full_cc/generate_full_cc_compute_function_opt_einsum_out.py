
def add_m0_n1_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) fully_connected terms."""

    # make an iterable out of the `opt_connected_path_list`
    optimized_einsum = iter(opt_connected_path_list)

    if ansatz.ground_state:
        R += h_args[(1, 0)]
    else:
        R += h_args[(1, 0)]
    return


def add_m0_n1_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) linked_disconnected terms."""

    # make an iterable out of the `opt_linked_path_list`
    optimized_einsum = iter(opt_linked_path_list)

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m0_n1_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) unlinked_disconnected terms."""

    # make an iterable out of the `opt_unlinked_path_list`
    optimized_einsum = iter(opt_unlinked_path_list)

    if ansatz.ground_state:
        if truncation.singles:
            R += next(optimized_einsum)(h_args[(0, 0)], t_args[(1, 0)])
    else:
        if truncation.singles:
            R += next(optimized_einsum)(h_args[(0, 0)], t_args[(1, 0)])
    return

