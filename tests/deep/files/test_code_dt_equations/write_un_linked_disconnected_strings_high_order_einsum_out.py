
def _order_5_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_unlinked_path_list):
    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.
    This means for order 5 we include terms such as (3, 2), (2, 2, 1)
    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `t_args` and 'dt_args'
    t_i, t_ij, t_ijk, t_ijkl, *unusedargs = t_args
    dt_i, dt_ij, dt_ijk, dt_ijkl, *unusedargs = dt_args
    # make an iterable out of the `opt_unlinked_path_list`
    optimized_einsum = iter(opt_unlinked_path_list)
    # Creating the 5th order return array
    un_linked_disconnected_terms = np.zeros((A, A, N, N, N, N, N), dtype=complex)
    # the (3, 2) term
    un_linked_disconnected_terms += 1/(factorial(2) * factorial(3) * factorial(2)) * (
        next(optimized_einsum)(dt_ij, t_ijk) +
        next(optimized_einsum)(t_ij, dt_ijk) +
        next(optimized_einsum)(dt_ijk, t_ij) +
        next(optimized_einsum)(t_ijk, dt_ij)
    )
    # the (2, 2, 1) term
    un_linked_disconnected_terms += 1/(factorial(3) * factorial(2) * factorial(2)) * (
        next(optimized_einsum)(dt_i, t_ij, t_ij) +
        next(optimized_einsum)(t_i, dt_ij, t_ij) +
        next(optimized_einsum)(t_i, t_ij, dt_ij) +
        next(optimized_einsum)(dt_ij, t_i, t_ij) +
        next(optimized_einsum)(t_ij, dt_i, t_ij) +
        next(optimized_einsum)(t_ij, t_i, dt_ij) +
        next(optimized_einsum)(dt_ij, t_ij, t_i) +
        next(optimized_einsum)(t_ij, dt_ij, t_i) +
        next(optimized_einsum)(t_ij, t_ij, dt_i)
    )

    return un_linked_disconnected_terms
