
# -------------- operator(name='', rank=0, m=0, n=0) TERMS -------------- #
def add_m0_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='', rank=0, m=0, n=0) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(0, 0)]

        if truncation.at_least_linear:
            if truncation.singles:
                R += np.einsum('aci, cbi -> ab', h_args[(0, 1)], t_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='', rank=0, m=0, n=0) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='', rank=0, m=0, n=0) unlinked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def add_m0_n1_fully_connected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) fully_connected terms."""

    if ansatz.ground_state:
        R += h_args[(1, 0)]
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n1_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n1_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='b', rank=1, m=0, n=1) unlinked_disconnected terms."""

    if ansatz.ground_state:
        if truncation.singles:
            R += np.einsum('ac, cbz -> abz', h_args[(0, 0)], t_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


# -------------- operator(name='d', rank=1, m=1, n=0) TERMS -------------- #
def add_m1_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='d', rank=1, m=1, n=0) fully_connected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m1_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='d', rank=1, m=1, n=0) linked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m1_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args):
    """Calculate the operator(name='d', rank=1, m=1, n=0) unlinked_disconnected terms."""

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return

