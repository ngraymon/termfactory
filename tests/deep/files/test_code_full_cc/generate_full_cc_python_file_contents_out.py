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

# ------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- OPTIMIZED PATHS FUNCTIONS ----------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #

# ----------------------------------------- INDIVIDUAL OPTIMIZED PATHS ----------------------------------------- #


# ------------ operator(name='', rank=0, m=0, n=0) OPTIMIZED PATHS ------------ #
def compute_m0_n0_fully_connected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the fully_connected terms."""

    fully_connected_opt_path_list = []

    if ansatz.ground_state:

        if truncation.at_least_linear:
            if truncation.singles:
                fully_connected_opt_path_list.append(oe.contract_expression((A, A, N), (A, A, N)))
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return fully_connected_opt_path_list


def compute_m0_n0_linked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the linked_disconnected terms."""

    linked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return linked_disconnected_opt_path_list


def compute_m0_n0_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the unlinked_disconnected terms."""

    unlinked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return unlinked_disconnected_opt_path_list

# --------------------------------------------------------------------------- #
# ------------------------- RANK  1 OPTIMIZED PATHS ------------------------- #
# --------------------------------------------------------------------------- #


# ------------ operator(name='b', rank=1, m=0, n=1) OPTIMIZED PATHS ------------ #
def compute_m0_n1_fully_connected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the fully_connected terms."""

    fully_connected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return fully_connected_opt_path_list


def compute_m0_n1_linked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the linked_disconnected terms."""

    linked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return linked_disconnected_opt_path_list


def compute_m0_n1_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the unlinked_disconnected terms."""

    unlinked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        if truncation.singles:
            unlinked_disconnected_opt_path_list.append(oe.contract_expression((A, A), (A, A, N)))
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return unlinked_disconnected_opt_path_list


# ------------ operator(name='d', rank=1, m=1, n=0) OPTIMIZED PATHS ------------ #
def compute_m1_n0_fully_connected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the fully_connected terms."""

    fully_connected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return fully_connected_opt_path_list


def compute_m1_n0_linked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the linked_disconnected terms."""

    linked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return linked_disconnected_opt_path_list


def compute_m1_n0_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation):
    """Calculate optimized einsum paths for the unlinked_disconnected terms."""

    unlinked_disconnected_opt_path_list = []

    if ansatz.ground_state:
        pass  # no valid terms here
    else:
        raise Exception('Hot Band amplitudes not implemented properly and have not been theoretically verified!')

    return unlinked_disconnected_opt_path_list


# ----------------------------------------- GROUPED BY PROJECTION OPERATOR ----------------------------------------- #
def compute_m0_n0_optimized_paths(A, N, ansatz, truncation):
    """Compute the optimized paths for this operator(name='', rank=0, m=0, n=0)."""
    truncation.confirm_at_least_singles()

    connected_opt_path_list = compute_m0_n0_fully_connected_optimized_paths(A, N, ansatz, truncation)
    linked_opt_path_list = compute_m0_n0_linked_disconnected_optimized_paths(A, N, ansatz, truncation)
    unlinked_opt_path_list = compute_m0_n0_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation)

    return_dict = {
        (0, 0): [connected_opt_path_list, linked_opt_path_list, unlinked_opt_path_list]
    }

    return return_dict


def compute_m0_n1_optimized_paths(A, N, ansatz, truncation):
    """Compute the optimized paths for this operator(name='b', rank=1, m=0, n=1)."""
    truncation.confirm_at_least_singles()

    connected_opt_path_list = compute_m0_n1_fully_connected_optimized_paths(A, N, ansatz, truncation)
    linked_opt_path_list = compute_m0_n1_linked_disconnected_optimized_paths(A, N, ansatz, truncation)
    unlinked_opt_path_list = compute_m0_n1_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation)

    return_dict = {
        (0, 1): [connected_opt_path_list, linked_opt_path_list, unlinked_opt_path_list]
    }

    return return_dict


def compute_m1_n0_optimized_paths(A, N, ansatz, truncation):
    """Compute the optimized paths for this operator(name='d', rank=1, m=1, n=0)."""
    truncation.confirm_at_least_singles()

    connected_opt_path_list = compute_m1_n0_fully_connected_optimized_paths(A, N, ansatz, truncation)
    linked_opt_path_list = compute_m1_n0_linked_disconnected_optimized_paths(A, N, ansatz, truncation)
    unlinked_opt_path_list = compute_m1_n0_unlinked_disconnected_optimized_paths(A, N, ansatz, truncation)

    return_dict = {
        (1, 0): [connected_opt_path_list, linked_opt_path_list, unlinked_opt_path_list]
    }

    return return_dict


# ----------------------------------------- MASTER OPTIMIZED PATH FUNCTION ----------------------------------------- #
def compute_all_optimized_paths(A, N, ansatz, truncation):
    """Return dictionary containing optimized contraction paths.
    Calculates all optimized paths for the `opt_einsum` calls up to
        a maximum order of m+n=1 for a projection operator P^m_n
    """
    all_opt_path_lists = []

    all_opt_path_lists[(0, 0)] = compute_m0_n0_optimized_paths(A, N, ansatz, truncation)[(0, 0)]
    all_opt_path_lists[(0, 1)] = compute_m0_n1_optimized_paths(A, N, ansatz, truncation)[(0, 1)]
    all_opt_path_lists[(1, 0)] = compute_m1_n0_optimized_paths(A, N, ansatz, truncation)[(1, 0)]

    return all_opt_path_lists


