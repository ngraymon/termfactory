
def add_m0_n1_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(1, 0)]

        if truncation.at_least_linear:
    else:
        R += h_args[(1, 0)]

        if truncation.at_least_linear:
    return


def add_m0_n1_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m0_n1_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_paths):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) unlinked_disconnected terms."""

    if ansatz.ground_state:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])

        if truncation.at_least_linear:
    else:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])

        if truncation.at_least_linear:
    return
