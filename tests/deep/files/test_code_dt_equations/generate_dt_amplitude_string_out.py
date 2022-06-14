# --------------------------------------------------------------------------- #
# ---------------------------- DEFAULT FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #

# ---------------------------- DISCONNECTED TERMS ---------------------------- #

def _order_1_linked_disconnected_terms(A, N, trunc, t_args, dt_args):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"
        "which requires a residual of at least 2nd order"
    )

def _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args):
    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.
    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)
    But not terms (5), (3, 2), (2, 2, 1)
    """
    # unpack the `t_args` and 'dt_args'
    t_i, *unusedargs = t_args
    dt_i, *unusedargs = dt_args
    # Creating the 2nd order return array
    linked_disconnected_terms = np.zeros((A, A, N, N), dtype=complex)
    # the (1, 1) term
    linked_disconnected_terms += 1/factorial(2) * (
        np.einsum('aci, cbj->abij', dt_i, t_i) +
        np.einsum('aci, cbj->abij', t_i, dt_i)
    )

    return linked_disconnected_terms

def _order_1_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a residual of at least 4th order"
    )

def _order_2_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a residual of at least 4th order"
    )

# ---------------------------- dt AMPLITUDES ---------------------------- #

def _calculate_order_1_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):
    """Calculate the derivative of the 1 t-amplitude for use in the calculation of the residuals."""
    # unpack the `w_args`
    w_i, *unusedargs = w_args
    # Calculate the 1st order residual
    residual = residual_equations.calculate_order_1_residual(A, N, trunc, h_args, w_args)
    # subtract the epsilon term (which is R_0)
    residual -= 1/factorial(1) * np.einsum('aci,cb->abi', w_i, epsilon)

    # subtract the disconnected terms
    if ansatz.VECI:
        pass  # veci does not include any disconnected terms
    elif ansatz.VE_MIXED:
        pass  # no linked disconnected terms for order < 2
    elif ansatz.VECC:
        pass  # no un-linked disconnected terms for order < 4

    # Symmetrize the residual operator
    dt_i = symmetrize_tensor(N, residual, order=1)
    return dt_i

def _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):
    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals."""
    # unpack the `w_args`
    w_i, w_ij, *unusedargs = w_args
    # Calculate the 2nd order residual
    residual = residual_equations.calculate_order_2_residual(A, N, trunc, h_args, w_args)
    # subtract the epsilon term (which is R_0)
    residual -= 1/factorial(2) * np.einsum('acij,cb->abij', w_ij, epsilon)

    # subtract the disconnected terms
    if ansatz.VECI:
        pass  # veci does not include any disconnected terms
    elif ansatz.VE_MIXED:
        residual -= _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args)
    elif ansatz.VECC:
        residual -= _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args)
        pass  # no un-linked disconnected terms for order < 4

    # Symmetrize the residual operator
    dt_ij = symmetrize_tensor(N, residual, order=2)
    return dt_ij

# ---------------------------- WRAPPER FUNCTIONS ---------------------------- #

def solve_singles_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):
    """Compute the change in the t_i term (singles)"""

    if not trunc.singles:
        raise Exception(
            "It appears that singles is not true, this cannot be."
            "Something went terribly wrong!!!"
        )
    dt_i = _calculate_order_1_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)
    return dt_i

def solve_doubles_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):
    """Compute the change in the t_ij term (doubles)"""

    if not trunc.doubles:
        raise Exception(
            "It appears that doubles is not true, this cannot be."
            "Something went terribly wrong!!!"
        )
    dt_ij = _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)
    return dt_ij

# --------------------------------------------------------------------------- #
# --------------------------- OPTIMIZED FUNCTIONS --------------------------- #
# --------------------------------------------------------------------------- #

# ---------------------------- DISCONNECTED TERMS ---------------------------- #

def _order_1_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_linked_path_list):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"
        "which requires a residual of at least 2nd order"
    )

def _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_linked_path_list):
    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.
    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1)
    But not terms (5), (3, 2), (2, 2, 1), (1, 1, 1, 1, 1)
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `t_args` and 'dt_args'
    t_i, *unusedargs = t_args
    dt_i, *unusedargs = dt_args
    # make an iterable out of the `opt_linked_path_list`
    optimized_einsum = iter(opt_linked_path_list)
    # Creating the 2nd order return array
    linked_disconnected_terms = np.zeros((A, A, N, N), dtype=complex)
    # the (1, 1) term
    linked_disconnected_terms += 1/factorial(2) * (
        next(optimized_einsum)(dt_i, t_i) +
        next(optimized_einsum)(t_i, dt_i)
    )

    return linked_disconnected_terms

def _order_1_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_unlinked_path_list):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a residual of at least 4th order"
    )

def _order_2_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_unlinked_path_list):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a residual of at least 4th order"
    )

# ---------------------------- dt AMPLITUDES ---------------------------- #

def _calculate_order_1_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_linked_path_list, opt_unlinked_path_list):
    """Calculate the derivative of the 1 t-amplitude for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `w_args`
    w_i, *unusedargs = w_args
    # Calculate the 1st order residual
    residual = residual_equations.calculate_order_1_residual(A, N, trunc, h_args, w_args)
    # subtract the epsilon term (which is R_0)
    residual -= 1/factorial(1) * opt_epsilon(w_i, epsilon)

    # subtract the disconnected terms
    if ansatz.VECI:
        pass  # veci does not include any disconnected terms
    elif ansatz.VE_MIXED:
        pass  # no linked disconnected terms for order < 2
    elif ansatz.VECC:
        pass  # no un-linked disconnected terms for order < 4

    # Symmetrize the residual operator
    dt_i = symmetrize_tensor(N, residual, order=1)
    return dt_i

def _calculate_order_2_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_linked_path_list, opt_unlinked_path_list):
    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `w_args`
    w_i, w_ij, *unusedargs = w_args
    # Calculate the 2nd order residual
    residual = residual_equations.calculate_order_2_residual(A, N, trunc, h_args, w_args)
    # subtract the epsilon term (which is R_0)
    residual -= 1/factorial(2) * opt_epsilon(w_ij, epsilon)

    # subtract the disconnected terms
    if ansatz.VECI:
        pass  # veci does not include any disconnected terms
    elif ansatz.VE_MIXED:
        residual -= _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_linked_path_list)
    elif ansatz.VECC:
        residual -= _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_linked_path_list)
        pass  # no un-linked disconnected terms for order < 4

    # Symmetrize the residual operator
    dt_ij = symmetrize_tensor(N, residual, order=2)
    return dt_ij

# ---------------------------- WRAPPER FUNCTIONS ---------------------------- #

def solve_singles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):
    """Compute the change in the t_i term (singles)"""

    if not trunc.singles:
        raise Exception(
            "It appears that singles is not true, this cannot be."
            "Something went terribly wrong!!!"
        )

    # unpack the opt_einsum path's
    opt_epsilon_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_list

    dt_i = _calculate_order_1_dt_amplitude_optimized(
        A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args,
        opt_epsilon_path_list[0], opt_linked_path_list[0], opt_unlinked_path_list[0]
    )
    return dt_i

def solve_doubles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):
    """Compute the change in the t_ij term (doubles)"""

    if not trunc.doubles:
        raise Exception(
            "It appears that doubles is not true, this cannot be."
            "Something went terribly wrong!!!"
        )

    # unpack the opt_einsum path's
    opt_epsilon_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_list

    dt_ij = _calculate_order_2_dt_amplitude_optimized(
        A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args,
        opt_epsilon_path_list[1], opt_linked_path_list[1], opt_unlinked_path_list[1]
    )
    return dt_ij

# ---------------------------- OPTIMIZED PATHS FUNCTION ---------------------------- #

def compute_optimized_paths(A, N, truncation):
    """Calculate all optimized paths for the `opt_einsum` calls."""

    epsilon_opt_paths = compute_optimized_epsilon_paths(A, N, truncation)
    linked_opt_paths = compute_optimized_linked_paths(A, N, truncation)
    unlinked_opt_paths = compute_optimized_unlinked_paths(A, N, truncation)

    all_opt_paths = [epsilon_opt_paths, linked_opt_paths, unlinked_opt_paths]

    return all_opt_paths


def compute_optimized_epsilon_paths(A, N, truncation):
    """Calculate optimized paths for the constant/epsilon einsum calls up to `highest_order`."""

    epsilon_path_list = []

    if truncation.singles:
        epsilon_path_list.append(oe.contract_expression('aci,cb->abi', (A, A, N), (A, A)))

    if truncation.doubles:
        epsilon_path_list.append(oe.contract_expression('acij,cb->abij', (A, A, N, N), (A, A)))

    return epsilon_path_list


def compute_optimized_linked_paths(A, N, truncation):
    """Calculate optimized paths for the linked-disconnected einsum calls up to `highest_order`."""

    order_1_list, order_2_list, order_3_list = [], [], []
    order_4_list, order_5_list, order_6_list = [], [], []

    if truncation.singles:
        order_2_list.extend([
            oe.contract_expression('aci, cbj->abij', (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cbj->abij', (A, A, N), (A, A, N)),
        ])

    return [order_1_list, order_2_list]


def compute_optimized_unlinked_paths(A, N, truncation):
    """Calculate optimized paths for the unlinked-disconnected einsum calls up to `highest_order`."""

    order_1_list, order_2_list, order_3_list = [], [], []
    order_4_list, order_5_list, order_6_list = [], [], []

    if not truncation.doubles:
        log.warning('Did not calculate optimized unlinked paths of the dt amplitudes')
        return [[], [], [], [], [], []]

    return [order_1_list, order_2_list, order_3_list]

