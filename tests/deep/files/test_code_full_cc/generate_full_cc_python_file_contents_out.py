# ------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- DEFAULT FUNCTIONS --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------- INDIVIDUAL TERMS --------------------------------------------- #


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


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude(A, N, ansatz, truncation, h_args, t_args):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A), dtype=complex)

    # add each of the terms
    add_m0_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    return R


def compute_m0_n1_amplitude(A, N, ansatz, truncation, h_args, t_args):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # add each of the terms
    add_m0_n1_fully_connected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n1_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    add_m0_n1_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    return R


def compute_m1_n0_amplitude(A, N, ansatz, truncation, h_args, t_args):
    """Compute the operator(name='d', rank=1, m=1, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # add each of the terms
    add_m1_n0_fully_connected_terms(R, ansatz, truncation, h_args, t_args)
    add_m1_n0_linked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    add_m1_n0_unlinked_disconnected_terms(R, ansatz, truncation, h_args, t_args)
    return R

# ------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- OPTIMIZED FUNCTIONS -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

# --------------------------------------------- INDIVIDUAL TERMS --------------------------------------------- #


# -------------- operator(name='', rank=0, m=0, n=0) TERMS -------------- #
def add_m0_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list):
    """Optimized calculation of the operator(name='', rank=0, m=0, n=0) fully_connected terms."""

    # make an iterable out of the `opt_connected_path_list`
    optimized_einsum = iter(opt_connected_path_list)

    if ansatz.ground_state:
        R += h_args[(0, 0)]

        if truncation.at_least_linear:
            if truncation.singles:
                R += next(optimized_einsum)(h_args[(0, 1)], t_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list):
    """Optimized calculation of the operator(name='', rank=0, m=0, n=0) linked_disconnected terms."""

    # make an iterable out of the `opt_linked_path_list`
    optimized_einsum = iter(opt_linked_path_list)

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list):
    """Optimized calculation of the operator(name='', rank=0, m=0, n=0) unlinked_disconnected terms."""

    # make an iterable out of the `opt_unlinked_path_list`
    optimized_einsum = iter(opt_unlinked_path_list)

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return

# --------------------------------------------------------------------------- #
# ---------------------------- RANK  1 FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #


# -------------- operator(name='b', rank=1, m=0, n=1) TERMS -------------- #
def add_m0_n1_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) fully_connected terms."""

    # make an iterable out of the `opt_connected_path_list`
    optimized_einsum = iter(opt_connected_path_list)

    if ansatz.ground_state:
        R += h_args[(1, 0)]
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n1_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) linked_disconnected terms."""

    # make an iterable out of the `opt_linked_path_list`
    optimized_einsum = iter(opt_linked_path_list)

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m0_n1_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list):
    """Optimized calculation of the operator(name='b', rank=1, m=0, n=1) unlinked_disconnected terms."""

    # make an iterable out of the `opt_unlinked_path_list`
    optimized_einsum = iter(opt_unlinked_path_list)

    if ansatz.ground_state:
        if truncation.singles:
            R += next(optimized_einsum)(h_args[(0, 0)], t_args[(1, 0)])
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


# -------------- operator(name='d', rank=1, m=1, n=0) TERMS -------------- #
def add_m1_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list):
    """Optimized calculation of the operator(name='d', rank=1, m=1, n=0) fully_connected terms."""

    # make an iterable out of the `opt_connected_path_list`
    optimized_einsum = iter(opt_connected_path_list)

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m1_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list):
    """Optimized calculation of the operator(name='d', rank=1, m=1, n=0) linked_disconnected terms."""

    # make an iterable out of the `opt_linked_path_list`
    optimized_einsum = iter(opt_linked_path_list)

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


def add_m1_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list):
    """Optimized calculation of the operator(name='d', rank=1, m=1, n=0) unlinked_disconnected terms."""

    # make an iterable out of the `opt_unlinked_path_list`
    optimized_einsum = iter(opt_unlinked_path_list)

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')
    return


# --------------------------------------------- RESIDUAL FUNCTIONS --------------------------------------------- #
def compute_m0_n0_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_path_lists):
    """Compute the operator(name='', rank=0, m=0, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A), dtype=complex)

    # unpack the optimized paths
    opt_connected_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_lists[(0, 0)]

    # add each of the terms
    add_m0_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list)
    add_m0_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list)
    add_m0_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list)
    return R


def compute_m0_n1_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_path_lists):
    """Compute the operator(name='b', rank=1, m=0, n=1) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # unpack the optimized paths
    opt_connected_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_lists[(0, 1)]

    # add each of the terms
    add_m0_n1_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list)
    add_m0_n1_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list)
    add_m0_n1_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list)
    return R


def compute_m1_n0_amplitude_optimized(A, N, ansatz, truncation, h_args, t_args, opt_path_lists):
    """Compute the operator(name='d', rank=1, m=1, n=0) amplitude."""
    truncation.confirm_at_least_singles()

    # the residual tensor
    R = np.zeros(shape=(A, A, N), dtype=complex)

    # unpack the optimized paths
    opt_connected_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_lists[(1, 0)]

    # add each of the terms
    add_m1_n0_fully_connected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_connected_path_list)
    add_m1_n0_linked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_linked_path_list)
    add_m1_n0_unlinked_disconnected_terms_optimized(R, ansatz, truncation, h_args, t_args, opt_unlinked_path_list)
    return R


# --------------------------------------------- OPTIMIZED PATHS FUNCTION --------------------------------------------- #