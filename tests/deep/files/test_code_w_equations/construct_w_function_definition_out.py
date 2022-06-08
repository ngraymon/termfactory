
def _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation):
    """Calculate the order 2 W operator for use in the calculation of the residuals."""
    # unpack the `t_args`
    t_i, t_ij, *unusedargs = t_args
