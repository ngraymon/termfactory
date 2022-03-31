# system imports
import os
import itertools as it
from collections import namedtuple

# third party imports

# local imports

truncation_maximums = {
    'maximum_hamiltonian_rank': 6,  # maximum_h_rank
    'maximum_coupled_cluster_rank': 6,  # maximum_cc_rank
    's_taylor_max_order': 6,
    'omega_max_order': 6,
    'max_residual_order': 8,
    'max_w_order': 8,
    'dt_order': 8,
}

################################################################################
"""
defines namedtuple's for the different truncation types
"""
################################################################################
"""
full cc type:

maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations

eT_z_t type:

maximum_h_rank, maximum_cc_rank, maximum_T_rank, eT_taylor_max_order, omega_max_order = truncations

"""
fcc_key_list = [
    'maximum_hamiltonian_rank',
    'maximum_coupled_cluster_rank',
    's_taylor_max_order',
    'omega_max_order'
]


fcc_truncations_namedtuple = namedtuple('fcc_truncations', fcc_key_list)

eT_z_t_key_list = [
    'maximum_hamiltonian_rank',
    'maximum_coupled_cluster_rank',
    'maximum_T_rank',
    'eT_taylor_max_order',
    'omega_max_order'
]
eT_z_t_truncations_namedtuple = namedtuple('eT_z_t_truncations', eT_z_t_key_list)

###################################################################
"""
Verify concept
"""
# below assumes truncations is type==dictionary


def __verify_keys(truncations, key_list):
    """ x """
    for key in key_list:
        assert key in truncations, f"Missing key, {key = :s} not in provided dictionary {truncations=:}"
        assert truncations[key] >= 1, "Truncations need to be positive integers"
        assert truncations[key] <= truncation_maximums[key], f"Key {key} is over the maximum of {truncation_maximums[key]}"


def _verify_fcc_truncations(truncations):
    """ x """
    __verify_keys(truncations, fcc_key_list)

    # add unpack function? like _unpack_fcc_truncations
    # unpack functions would call its respective _verify_x function
    maximum_hamiltonian_rank = truncations["maximum_hamiltonian_rank"]
    maximum_coupled_cluster_rank = truncations["maximum_coupled_cluster_rank"]
    s_taylor_max_order = truncations["s_taylor_max_order"]
    omega_max_order = truncations["omega_max_order"]


def _verify_eT_z_t_truncations(truncations):
    """ x """
    __verify_keys(truncations, eT_z_t_key_list)

    # add unpack function? like _unpack_eT_z_t_truncations
    maximum_hamiltonian_rank = truncations["maximum_hamiltonian_rank"]
    maximum_coupled_cluster_rank = truncations["maximum_coupled_cluster_rank"]
    maximum_T_rank = truncations["maximum_T_rank"]
    eT_taylor_max_order = truncations["eT_taylor_max_order"]
    omega_max_order = truncations["omega_max_order "]


# test={
#     'maximum_hamiltonian_rank': 1,  # maximum_h_rank
#     'maximum_coupled_cluster_rank': 1,  # maximum_cc_rank
#     's_taylor_max_order': 1,
#     'omega_max_order': 1,
# }

# test2={
#     'maximum_hamiltonian_rank': 1,  # maximum_h_rank
#     'maximum_coupled_cluster_rank': 1,  # maximum_cc_rank
#     's_taylor_max_order': 1,
#     'omega_max_order': 1,
# }
# print(test[1])
# _verify_fcc_truncations(test)
