
def _add_order_3_vemx_contributions(W_3, t_args, truncation):
    """Calculate the order 3 VECI/CC (mixed) contributions to the W operator
    for use in the calculation of the residuals.
    """
    # unpack the `t_args`
    t_i, t_ij, *unusedargs = t_args
    # DOUBLES contribution
    if truncation.doubles:
        W_3 += 1/(factorial(2) * factorial(2)) * (
            np.einsum('aci, cbjk->abijk', t_i, t_ij) +
            np.einsum('acij, cbk->abijk', t_ij, t_i)
        )
    # SINGLES contribution
    W_3 += 1/factorial(3) * (np.einsum('aci, cdj, dbk->abijk', t_i, t_i, t_i))
    return
