
def _calculate_order_5_w_operator(A, N, t_args, ansatz, truncation):
    """Calculate the order 5 W operator for use in the calculation of the residuals."""
    # unpack the `t_args`
    t_i, t_ij, t_ijk, t_ijkl, t_ijklm, *unusedargs = t_args
    # Creating the 5th order W operator
    W_5 = np.zeros((A, A, N, N, N, N, N), dtype=complex)

    # add the VECI contribution
    if truncation.quintuples:
        W_5 += 1/factorial(5) * t_ijklm
    if ansatz.VE_MIXED:
        _add_order_5_vemx_contributions(W_5, t_args, truncation)
    elif ansatz.VECC:
        _add_order_5_vemx_contributions(W_5, t_args, truncation)
        _add_order_5_vecc_contributions(W_5, t_args, truncation)

    # Symmetrize the W operator
    symmetric_w = symmetrize_tensor(N, W_5, order=5)
    return symmetric_w
