
def solve_doubles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):
    """Compute the change in the t_ij term (doubles)"""

    if not trunc.doubles:
        raise Exception(
            "It appears that doubles is not true, this cannot be."
            "Something went terribly wrong!!!"
        )

    # unpack the opt_einsum path's
    opt_epsilon_path_list, opt_linked_path_list, opt_unlinked_path_list = opt_path_list

    dt_ij = _calculate_order_2_dt_amplitude_optimized(
        A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args,
        opt_epsilon_path_list[1], opt_linked_path_list[1], opt_unlinked_path_list[1]
    )
    return dt_ij
