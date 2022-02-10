
def _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation):
    """Calculate the order 1 W operator for use in the calculation of the residuals."""
    # unpack the `t_args`
    t_i, *unusedargs = t_args
    # Creating the 1st order W operator
    W_1 = np.zeros((A, A, N), dtype=complex)
    # Singles contribution
    W_1 += t_i
    return W_1
