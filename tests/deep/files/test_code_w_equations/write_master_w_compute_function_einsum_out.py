
def compute_w_operators_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths, vecc_optimized_paths):
    """Compute a number of W operators depending on the level of truncation."""

    if not truncation.singles:
        raise Exception(
            "It appears that `singles` is not true, this cannot be.\n"
            "Something went terribly wrong!!!\n\n"
            f"{truncation}\n"
        )

    w_1 = _calculate_order_1_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[0], vecc_optimized_paths[0])
    w_2 = _calculate_order_2_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[1], vecc_optimized_paths[1])
    w_3 = _calculate_order_3_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[2], vecc_optimized_paths[2])

    if not truncation.doubles:
        return w_1, w_2, w_3, None, None, None
    else:
        w_4 = _calculate_order_4_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[3], vecc_optimized_paths[3])

    if not truncation.triples:
        return w_1, w_2, w_3, w_4, None, None
    else:
        w_5 = _calculate_order_5_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[4], vecc_optimized_paths[4])

    if not truncation.quadruples:
        return w_1, w_2, w_3, w_4, w_5, None
    else:
        w_6 = _calculate_order_6_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[5], vecc_optimized_paths[5])

    if not truncation.quintuples:
        return w_1, w_2, w_3, w_4, w_5, w_6
    else:
        raise Exception(
            "Attempting to calculate W^7 operator (quintuples)\n"
            "This is currently not implemented!!\n"
        )
