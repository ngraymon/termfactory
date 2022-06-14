
def compute_optimized_linked_paths(A, N, truncation):
    """Calculate optimized paths for the linked-disconnected einsum calls up to `highest_order`."""

    order_1_list, order_2_list, order_3_list = [], [], []
    order_4_list, order_5_list, order_6_list = [], [], []

    if truncation.singles:
        order_2_list.extend([
            oe.contract_expression('aci, cbj->abij', (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cbj->abij', (A, A, N), (A, A, N)),
        ])

    if truncation.doubles:
        order_3_list.extend([
            oe.contract_expression('aci, cbjk->abijk', (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cbjk->abijk', (A, A, N), (A, A, N, N)),
            oe.contract_expression('acij, cbk->abijk', (A, A, N, N), (A, A, N)),
            oe.contract_expression('acij, cbk->abijk', (A, A, N, N), (A, A, N)),
        ])

    if truncation.singles:
        order_3_list.extend([
            oe.contract_expression('aci, cdj, dbk->abijk', (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dbk->abijk', (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dbk->abijk', (A, A, N), (A, A, N), (A, A, N)),
        ])

    if truncation.triples:
        order_4_list.extend([
            oe.contract_expression('aci, cbjkl->abijkl', (A, A, N), (A, A, N, N, N)),
            oe.contract_expression('aci, cbjkl->abijkl', (A, A, N), (A, A, N, N, N)),
            oe.contract_expression('acijk, cbl->abijkl', (A, A, N, N, N), (A, A, N)),
            oe.contract_expression('acijk, cbl->abijkl', (A, A, N, N, N), (A, A, N)),
        ])

    if truncation.doubles:
        order_4_list.extend([
            oe.contract_expression('aci, cdj, dbkl->abijkl', (A, A, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cdj, dbkl->abijkl', (A, A, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cdj, dbkl->abijkl', (A, A, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cdjk, dbl->abijkl', (A, A, N), (A, A, N, N), (A, A, N)),
            oe.contract_expression('aci, cdjk, dbl->abijkl', (A, A, N), (A, A, N, N), (A, A, N)),
            oe.contract_expression('aci, cdjk, dbl->abijkl', (A, A, N), (A, A, N, N), (A, A, N)),
            oe.contract_expression('acij, cdk, dbl->abijkl', (A, A, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acij, cdk, dbl->abijkl', (A, A, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acij, cdk, dbl->abijkl', (A, A, N, N), (A, A, N), (A, A, N)),
        ])

    if truncation.singles:
        order_4_list.extend([
            oe.contract_expression('aci, cdj, dek, ebl->abijkl', (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dek, ebl->abijkl', (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dek, ebl->abijkl', (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dek, ebl->abijkl', (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
        ])

    if truncation.quadruples:
        order_5_list.extend([
            oe.contract_expression('aci, cbjklm->abijklm', (A, A, N), (A, A, N, N, N, N)),
            oe.contract_expression('aci, cbjklm->abijklm', (A, A, N), (A, A, N, N, N, N)),
            oe.contract_expression('acijkl, cbm->abijklm', (A, A, N, N, N, N), (A, A, N)),
            oe.contract_expression('acijkl, cbm->abijklm', (A, A, N, N, N, N), (A, A, N)),
        ])

    if truncation.triples:
        order_5_list.extend([
            oe.contract_expression('aci, cdj, dbklm->abijklm', (A, A, N), (A, A, N), (A, A, N, N, N)),
            oe.contract_expression('aci, cdj, dbklm->abijklm', (A, A, N), (A, A, N), (A, A, N, N, N)),
            oe.contract_expression('aci, cdj, dbklm->abijklm', (A, A, N), (A, A, N), (A, A, N, N, N)),
            oe.contract_expression('aci, cdjkl, dbm->abijklm', (A, A, N), (A, A, N, N, N), (A, A, N)),
            oe.contract_expression('aci, cdjkl, dbm->abijklm', (A, A, N), (A, A, N, N, N), (A, A, N)),
            oe.contract_expression('aci, cdjkl, dbm->abijklm', (A, A, N), (A, A, N, N, N), (A, A, N)),
            oe.contract_expression('acijk, cdl, dbm->abijklm', (A, A, N, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acijk, cdl, dbm->abijklm', (A, A, N, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acijk, cdl, dbm->abijklm', (A, A, N, N, N), (A, A, N), (A, A, N)),
        ])

    if truncation.doubles:
        order_5_list.extend([
            oe.contract_expression('aci, cdj, dek, eblm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cdj, dek, eblm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cdj, dek, eblm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cdj, dek, eblm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N, N)),
            oe.contract_expression('aci, cdj, dekl, ebm->abijklm', (A, A, N), (A, A, N), (A, A, N, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dekl, ebm->abijklm', (A, A, N), (A, A, N), (A, A, N, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dekl, ebm->abijklm', (A, A, N), (A, A, N), (A, A, N, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dekl, ebm->abijklm', (A, A, N), (A, A, N), (A, A, N, N), (A, A, N)),
            oe.contract_expression('aci, cdjk, del, ebm->abijklm', (A, A, N), (A, A, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdjk, del, ebm->abijklm', (A, A, N), (A, A, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdjk, del, ebm->abijklm', (A, A, N), (A, A, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdjk, del, ebm->abijklm', (A, A, N), (A, A, N, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acij, cdk, del, ebm->abijklm', (A, A, N, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acij, cdk, del, ebm->abijklm', (A, A, N, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acij, cdk, del, ebm->abijklm', (A, A, N, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('acij, cdk, del, ebm->abijklm', (A, A, N, N), (A, A, N), (A, A, N), (A, A, N)),
        ])

    if truncation.singles:
        order_5_list.extend([
            oe.contract_expression('aci, cdj, dek, efl, fbm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dek, efl, fbm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dek, efl, fbm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dek, efl, fbm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
            oe.contract_expression('aci, cdj, dek, efl, fbm->abijklm', (A, A, N), (A, A, N), (A, A, N), (A, A, N), (A, A, N)),
        ])

    return [order_1_list, order_2_list, order_3_list, order_4_list, order_5_list]
