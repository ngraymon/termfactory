# --------------------------------------------------------------------------- #
# ---------------------------- DEFAULT FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #

# ---------------------------- VECI/CC CONTRIBUTIONS ---------------------------- #

def _add_order_1_vemx_contributions(W_1, t_args, truncation):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"
        "which requires a W operator of at least 2nd order"
    )

def _add_order_2_vemx_contributions(W_2, t_args, truncation):
    """Calculate the order 2 VECI/CC (mixed) contributions to the W operator
    for use in the calculation of the residuals.
    """
    # unpack the `t_args`
    t_i, *unusedargs = t_args
    # SINGLES contribution
    W_2 += 1/factorial(2) * (np.einsum('aci, cbj->abij', t_i, t_i))
    return

# ---------------------------- VECC CONTRIBUTIONS ---------------------------- #

def _add_order_1_vecc_contributions(W_1, t_args, truncation):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a W operator of at least 4th order"
    )

def _add_order_2_vecc_contributions(W_2, t_args, truncation):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a W operator of at least 4th order"
    )

# ---------------------------- W OPERATOR FUNCTIONS ---------------------------- #

def _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation):
    """Calculate the order 1 W operator for use in the calculation of the residuals."""
    # unpack the `t_args`
    t_i, *unusedargs = t_args
    # Creating the 1st order W operator
    W_1 = np.zeros((A, A, N), dtype=complex)
    # Singles contribution
    W_1 += t_i
    return W_1

def _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation):
    """Calculate the order 2 W operator for use in the calculation of the residuals."""
    # unpack the `t_args`
    t_i, t_ij, *unusedargs = t_args
    # Creating the 2nd order W operator
    W_2 = np.zeros((A, A, N, N), dtype=complex)

    # add the VECI contribution
    if truncation.doubles:
        W_2 += 1/factorial(2) * t_ij
    if ansatz.VE_MIXED:
        _add_order_2_vemx_contributions(W_2, t_args, truncation)
    elif ansatz.VECC:
        _add_order_2_vemx_contributions(W_2, t_args, truncation)
        pass  # no VECC contributions for order < 4

    # Symmetrize the W operator
    symmetric_w = symmetrize_tensor(N, W_2, order=2)
    return symmetric_w

def compute_w_operators(A, N, t_args, ansatz, truncation):
    """Compute a number of W operators depending on the level of truncation."""

    if not truncation.singles:
        raise Exception(
            "It appears that `singles` is not true, this cannot be.\n"
            "Something went terribly wrong!!!\n\n"
            f"{truncation}\n"
        )

    w_1 = _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation)
    w_2 = _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation)
    w_3 = _calculate_order_3_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.doubles:
        return w_1, w_2, w_3, None, None, None
    else:
        w_4 = _calculate_order_4_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.triples:
        return w_1, w_2, w_3, w_4, None, None
    else:
        w_5 = _calculate_order_5_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.quadruples:
        return w_1, w_2, w_3, w_4, w_5, None
    else:
        w_6 = _calculate_order_6_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.quintuples:
        return w_1, w_2, w_3, w_4, w_5, w_6
    else:
        raise Exception(
            "Attempting to calculate W^7 operator (quintuples)\n"
            "This is currently not implemented!!\n"
        )

# --------------------------------------------------------------------------- #
# --------------------------- OPTIMIZED FUNCTIONS --------------------------- #
# --------------------------------------------------------------------------- #

# ---------------------------- VECI/CC CONTRIBUTIONS ---------------------------- #

def _add_order_1_vemx_contributions_optimized(W_1, t_args, truncation, opt_path_list):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"
        "which requires a W operator of at least 2nd order"
    )

def _add_order_2_vemx_contributions_optimized(W_2, t_args, truncation, opt_path_list):
    """Calculate the order 2 VECI/CC (mixed) contributions to the W operator
    for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `t_args`
    t_i, *unusedargs = t_args
    # make an iterable out of the `opt_path_list`
    optimized_einsum = iter(opt_path_list)
    # SINGLES contribution
    W_2 += 1/factorial(2) * (next(optimized_einsum)(t_i, t_i))
    return

# ---------------------------- VECC CONTRIBUTIONS ---------------------------- #

def _add_order_1_vecc_contributions_optimized(W_1, t_args, truncation, opt_path_list):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a W operator of at least 4th order"
    )

def _add_order_2_vecc_contributions_optimized(W_2, t_args, truncation, opt_path_list):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a W operator of at least 4th order"
    )

# ---------------------------- W OPERATOR FUNCTIONS ---------------------------- #

def _calculate_order_1_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_opt_path_list, vecc_opt_path_list):
    """Calculate the order 1 W operator for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `t_args`
    t_i, *unusedargs = t_args
    # Creating the 1st order W operator
    W_1 = np.zeros((A, A, N), dtype=complex)
    # Singles contribution
    W_1 += t_i
    return W_1

def _calculate_order_2_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_opt_path_list, vecc_opt_path_list):
    """Calculate the order 2 W operator for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `t_args`
    t_i, t_ij, *unusedargs = t_args
    # Creating the 2nd order W operator
    W_2 = np.zeros((A, A, N, N), dtype=complex)

    # add the VECI contribution
    if truncation.doubles:
        W_2 += 1/factorial(2) * t_ij
    if ansatz.VE_MIXED:
        _add_order_2_vemx_contributions_optimized(W_2, t_args, truncation, vemx_opt_path_list)
    elif ansatz.VECC:
        _add_order_2_vemx_contributions_optimized(W_2, t_args, truncation, vemx_opt_path_list)
        pass  # no VECC contributions for order < 4

    # Symmetrize the W operator
    symmetric_w = symmetrize_tensor(N, W_2, order=2)
    return symmetric_w

def compute_w_operators_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths, vecc_optimized_paths):
    """Compute a number of W operators depending on the level of truncation."""

    if not truncation.singles:
        raise Exception(
            "It appears that `singles` is not true, this cannot be.\n"
            "Something went terribly wrong!!!\n\n"
            f"{truncation}\n"
        )

    w_1 = _calculate_order_1_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[0], vecc_optimized_paths[0])
    w_2 = _calculate_order_2_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[1], vecc_optimized_paths[1])
    w_3 = _calculate_order_3_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[2], vecc_optimized_paths[2])

    if not truncation.doubles:
        return w_1, w_2, w_3, None, None, None
    else:
        w_4 = _calculate_order_4_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[3], vecc_optimized_paths[3])

    if not truncation.triples:
        return w_1, w_2, w_3, w_4, None, None
    else:
        w_5 = _calculate_order_5_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[4], vecc_optimized_paths[4])

    if not truncation.quadruples:
        return w_1, w_2, w_3, w_4, w_5, None
    else:
        w_6 = _calculate_order_6_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[5], vecc_optimized_paths[5])

    if not truncation.quintuples:
        return w_1, w_2, w_3, w_4, w_5, w_6
    else:
        raise Exception(
            "Attempting to calculate W^7 operator (quintuples)\n"
            "This is currently not implemented!!\n"
        )


# ---------------------------- OPTIMIZED PATHS FUNCTION ---------------------------- #

def compute_optimized_vemx_paths(A, N, truncation):
    """Calculate optimized paths for the VECI/CC (mixed) einsum calls up to `highest_order`."""

    order_2_list, order_3_list = [], []
    order_4_list, order_5_list, order_6_list = [], [], []

    if truncation.singles:
        order_2_list.extend([
            oe.contract_expression('aci, cbj->abij', (A, A, N), (A, A, N)),
        ])


    return [[], order_2_list]


def compute_optimized_vecc_paths(A, N, truncation):
    """Calculate optimized paths for the VECC einsum calls up to `highest_order`."""

    order_4_list, order_5_list, order_6_list = [], [], []

    if not truncation.doubles:
        log.warning('Did not calculate optimized VECC paths of the dt amplitudes')
        return [[], [], [], [], [], []]


    return [[], [], []]

