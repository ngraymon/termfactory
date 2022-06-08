
def _order_5_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):
    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.
    This means for order 5 we include terms such as (3, 2), (2, 2, 1)
    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)
    """
    # unpack the `t_args` and 'dt_args'
    t_i, t_ij, t_ijk, t_ijkl, *unusedargs = t_args
    dt_i, dt_ij, dt_ijk, dt_ijkl, *unusedargs = dt_args
    # Creating the 5th order return array
    un_linked_disconnected_terms = np.zeros((A, A, N, N, N, N, N), dtype=complex)
    # the (3, 2) term
    un_linked_disconnected_terms += 1/(factorial(2) * factorial(3) * factorial(2)) * (
        np.einsum('acij, cbklm->abijklm', dt_ij, t_ijk) +
        np.einsum('acij, cbklm->abijklm', t_ij, dt_ijk) +
        np.einsum('acijk, cblm->abijklm', dt_ijk, t_ij) +
        np.einsum('acijk, cblm->abijklm', t_ijk, dt_ij)
    )
    # the (2, 2, 1) term
    un_linked_disconnected_terms += 1/(factorial(3) * factorial(2) * factorial(2)) * (
        np.einsum('aci, cdjk, dblm->abijklm', dt_i, t_ij, t_ij) +
        np.einsum('aci, cdjk, dblm->abijklm', t_i, dt_ij, t_ij) +
        np.einsum('aci, cdjk, dblm->abijklm', t_i, t_ij, dt_ij) +
        np.einsum('acij, cdk, dblm->abijklm', dt_ij, t_i, t_ij) +
        np.einsum('acij, cdk, dblm->abijklm', t_ij, dt_i, t_ij) +
        np.einsum('acij, cdk, dblm->abijklm', t_ij, t_i, dt_ij) +
        np.einsum('acij, cdkl, dbm->abijklm', dt_ij, t_ij, t_i) +
        np.einsum('acij, cdkl, dbm->abijklm', t_ij, dt_ij, t_i) +
        np.einsum('acij, cdkl, dbm->abijklm', t_ij, t_ij, dt_i)
    )

    return un_linked_disconnected_terms
