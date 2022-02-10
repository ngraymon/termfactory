
def compute_optimized_vecc_paths(A, N, truncation):
    """Calculate optimized paths for the VECC einsum calls up to `highest_order`."""

    order_4_list, order_5_list, order_6_list = [], [], []

    if not truncation.doubles:
        log.warning('Did not calculate optimized VECC paths of the dt amplitudes')
        return [[], [], [], [], [], []]


    return [[], [], []]
