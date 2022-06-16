
def add_m0_n1_fully_connected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(1, 0)]
    else:
        R += h_args[(1, 0)]
    return


def add_m0_n1_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        pass  # no valid terms here
    return


def add_m0_n1_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) unlinked_disconnected terms."""

    if ansatz.ground_state:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])
    else:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])
    return

