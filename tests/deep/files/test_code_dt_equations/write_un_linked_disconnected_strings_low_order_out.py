
def _order_2_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"
        "which requires a residual of at least 4th order"
    )
