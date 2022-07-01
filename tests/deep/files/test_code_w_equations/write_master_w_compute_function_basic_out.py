
def compute_w_operators(A, N, t_args, ansatz, truncation):
    """Compute a number of W operators depending on the level of truncation."""

    if not truncation.singles:
        raise Exception(
            "It appears that `singles` is not true, this cannot be.\n"
            "Something went terribly wrong!!!\n\n"
            f"{truncation}\n"
        )

    w_1 = _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation)
    w_2 = _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation)
    w_3 = _calculate_order_3_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.doubles:
        return w_1, w_2, w_3, None, None, None
    else:
        w_4 = _calculate_order_4_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.triples:
        return w_1, w_2, w_3, w_4, None, None
    else:
        w_5 = _calculate_order_5_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.quadruples:
        return w_1, w_2, w_3, w_4, w_5, None
    else:
        w_6 = _calculate_order_6_w_operator(A, N, t_args, ansatz, truncation)

    if not truncation.quintuples:
        return w_1, w_2, w_3, w_4, w_5, w_6
    else:
        raise NotImplementedError(
            "Attempting to calculate W^7 operator (quintuples)\n"
            "This is currently not implemented!!\n"
        )
