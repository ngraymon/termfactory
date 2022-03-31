
def solve_doubles_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):
    """Compute the change in the t_ij term (doubles)"""

    if not trunc.doubles:
        raise Exception(
            "It appears that doubles is not true, this cannot be."
            "Something went terribly wrong!!!"
        )
    dt_ij = _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)
    return dt_ij
