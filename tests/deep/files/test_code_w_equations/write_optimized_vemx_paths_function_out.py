
def compute_optimized_vemx_paths(A, N, truncation):
    """Calculate optimized paths for the VECI/CC (mixed) einsum calls up to `highest_order`."""

    order_2_list, order_3_list = [], []
    order_4_list, order_5_list, order_6_list = [], [], []

    if truncation.singles:
        order_2_list.extend([
            oe.contract_expression('aci, cbj->abij', (A, A, N), (A, A, N)),
        ])


    return [[], order_2_list]
