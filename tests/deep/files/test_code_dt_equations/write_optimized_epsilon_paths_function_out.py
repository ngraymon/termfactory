
def compute_optimized_epsilon_paths(A, N, truncation):
    """Calculate optimized paths for the constant/epsilon einsum calls up to `highest_order`."""

    epsilon_path_list = []

    if truncation.singles:
        epsilon_path_list.append(oe.contract_expression('aci,cb->abi', (A, A, N), (A, A)))

    if truncation.doubles:
        epsilon_path_list.append(oe.contract_expression('acij,cb->abij', (A, A, N, N), (A, A)))

    if truncation.triples:
        epsilon_path_list.append(oe.contract_expression('acijk,cb->abijk', (A, A, N, N, N), (A, A)))

    if truncation.quadruples:
        epsilon_path_list.append(oe.contract_expression('acijkl,cb->abijkl', (A, A, N, N, N, N), (A, A)))

    if truncation.quintuples:
        epsilon_path_list.append(oe.contract_expression('acijklm,cb->abijklm', (A, A, N, N, N, N, N), (A, A)))

    return epsilon_path_list
