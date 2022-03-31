
def _add_order_5_vecc_contributions(W_5, t_args, truncation):
    """Calculate the order 5 VECC contributions to the W operator
    for use in the calculation of the residuals.
    """
    # unpack the `t_args`
    t_i, t_ij, t_ijk, *unusedargs = t_args
    # TRIPLES contribution
    if truncation.triples:
        W_5 += 1/(factorial(2) * factorial(3) * factorial(2)) * (
            np.einsum('acij, cbklm->abijklm', t_ij, t_ijk) +
            np.einsum('acijk, cblm->abijklm', t_ijk, t_ij)
        )
    # DOUBLES contribution
    if truncation.doubles:
        W_5 += 1/(factorial(3) * factorial(2) * factorial(2)) * (
            np.einsum('aci, cdjk, dblm->abijklm', t_i, t_ij, t_ij) +
            np.einsum('acij, cdk, dblm->abijklm', t_ij, t_i, t_ij) +
            np.einsum('acij, cdkl, dbm->abijklm', t_ij, t_ij, t_i)
        )
    return
