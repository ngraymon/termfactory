
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
