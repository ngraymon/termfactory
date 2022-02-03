
def compute_optimized_vecc_paths(A, N, truncation):
    """Calculate optimized paths for the VECC einsum calls up to `highest_order`."""

    order_4_list, order_5_list, order_6_list = [], [], []

    if not truncation.doubles:
        log.warning('Did not calculate optimized VECC paths of the dt amplitudes')
        return [[], [], [], [], [], []]

    if truncation.doubles:
        order_4_list.extend([
            oe.contract_expression('acij, cbkl->abijkl', (A, A, N, N), (A, A, N, N)),
        ])

    if truncation.triples:
        order_5_list.extend([
            oe.contract_expression('acij, cbklm->abijklm', (A, A, N, N), (A, A, N, N, N)),
            oe.contract_expression('acijk, cblm->abijklm', (A, A, N, N, N), (A, A, N, N)),
        ])

    if truncation.doubles:
        order_5_list.extend([
            oe.contract_expression('aci, cdjk, dblm->abijklm', (A, A, N), (A, A, N, N), (A, A, N, N)),
            oe.contract_expression('acij, cdk, dblm->abijklm', (A, A, N, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('acij, cdkl, dbm->abijklm', (A, A, N, N), (A, A, N, N), (A, A, N)),
        ])


    return [[], [], [], order_4_list, order_5_list]
