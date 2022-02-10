
def _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):
    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals."""
    # unpack the `w_args`
    w_i, w_ij, *unusedargs = w_args
