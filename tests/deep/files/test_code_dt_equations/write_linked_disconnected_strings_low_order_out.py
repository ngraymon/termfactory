
def _order_1_linked_disconnected_terms(A, N, trunc, t_args, dt_args):
    """Exists for error checking."""
    raise Exception(
        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"
        "which requires a residual of at least 2nd order"
    )
