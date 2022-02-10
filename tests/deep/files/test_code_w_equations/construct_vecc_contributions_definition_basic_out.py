
def _add_order_2_vecc_contributions(W_2, t_args, truncation):
    """Calculate the order 2 VECC contributions to the W operator
    for use in the calculation of the residuals.
    """
    # unpack the `t_args`
    *unusedargs = t_args
