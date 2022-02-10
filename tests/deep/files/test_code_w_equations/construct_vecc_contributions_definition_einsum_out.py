
def _add_order_2_vecc_contributions_optimized(W_2, t_args, truncation, opt_path_list):
    """Calculate the order 2 VECC contributions to the W operator
    "for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `t_args`
    *unusedargs = t_args
    # make an iterable out of the `opt_path_list`
    optimized_einsum = iter(opt_path_list)
