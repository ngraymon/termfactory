
def _add_order_3_vemx_contributions_optimized(W_3, t_args, truncation, opt_path_list):
    """Calculate the order 3 VECI/CC (mixed) contributions to the W operator
    for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `t_args`
    t_i, t_ij, *unusedargs = t_args
    # make an iterable out of the `opt_path_list`
    optimized_einsum = iter(opt_path_list)
