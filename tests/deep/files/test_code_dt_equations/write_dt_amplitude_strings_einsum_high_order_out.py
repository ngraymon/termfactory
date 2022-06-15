
def _calculate_order_5_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_linked_path_list, opt_unlinked_path_list):
    """Calculate the derivative of the 5 t-amplitude for use in the calculation of the residuals.
    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.
    """
    # unpack the `w_args`
    w_i, w_ij, w_ijk, w_ijkl, w_ijklm, *unusedargs = w_args
    # Calculate the 5th order residual
    residual = residual_equations.calculate_order_5_residual(A, N, trunc, h_args, w_args)
    # subtract the epsilon term (which is R_0)
    residual -= 1/factorial(5) * opt_epsilon(w_ijklm, epsilon)

    # subtract the disconnected terms
    if ansatz.VECI:
        pass  # veci does not include any disconnected terms
    elif ansatz.VE_MIXED:
        residual -= _order_5_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_linked_path_list)
    elif ansatz.VECC:
        residual -= _order_5_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_linked_path_list)
        residual -= _order_5_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_unlinked_path_list)

    # Symmetrize the residual operator
    dt_ijklm = symmetrize_tensor(N, residual, order=5)
    return dt_ijklm
