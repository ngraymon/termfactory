
def _calculate_order_3_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):
    """Calculate the derivative of the 3 t-amplitude for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `w_args`
    w_i, w_ij, w_ijk, *unusedargs = w_args
    # make an iterable out of the `opt_path_list`
    optimized_einsum = iter(opt_path_list)
