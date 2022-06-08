
def _add_order_3_vemx_contributions(W_3, t_args, truncation):
    """Calculate the order 3 VECI/CC (mixed) contributions to the W operator
    for use in the calculation of the residuals.
    """
    # unpack the `t_args`
    t_i, t_ij, *unusedargs = t_args
