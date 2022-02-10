
def solve_doubles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):
    """Compute the change in the t_ij term (doubles)"""

    if not trunc.doubles:
        raise Exception(
            "It appears that doubles is not true, this cannot be."
            "Something went terribly wrong!!!"
        )
    dt_ij = _calculate_order_2_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list)
    return dt_ij
