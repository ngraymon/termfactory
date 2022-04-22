# system imports
import os
import copy
import json
import itertools as it
from collections import namedtuple
from enum import Enum
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
eT_z_t_key_list = [
    'maximum_hamiltonian_rank',
    'maximum_coupled_cluster_rank',
    'maximum_T_rank',
    'eT_taylor_max_order',
    'omega_max_order'
]
fcc_truncations_namedtuple = namedtuple('fcc_truncations', fcc_key_list)
eT_z_t_truncations_namedtuple = namedtuple('eT_z_t_truncations', eT_z_t_key_list)


###################################################################
"""Verify concept
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
    # fcc_nums=truncations.fcc_key_list
    print(truncations)
    
    print(truncations.H)
    # t_dict = repr(truncations.values.values)
    
    # maximum_hamiltonian_rank = truncations["maximum_hamiltonian_rank"]
    # maximum_coupled_cluster_rank = truncations["maximum_coupled_cluster_rank"]
    # s_taylor_max_order = truncations["s_taylor_max_order"]
    # omega_max_order = truncations["omega_max_order"]
    print(t_dict)
    # print(truncations["max_h_rank"])
    # __verify_keys(t_dict, fcc_key_list)


def _verify_eT_z_t_truncations(truncations):
    """ x """
    
    # eT_dct = {i.name: i.value for i in eT_trunc}
    # add unpack function? like _unpack_eT_z_t_truncations
    maximum_hamiltonian_rank = truncations["maximum_hamiltonian_rank"]
    maximum_coupled_cluster_rank = truncations["maximum_coupled_cluster_rank"]
    maximum_T_rank = truncations["maximum_T_rank"]
    eT_taylor_max_order = truncations["eT_taylor_max_order"]
    omega_max_order = truncations["omega_max_order"]

    __verify_keys(truncations, eT_z_t_key_list)

################################################################################
#                              Save / Load JSON                                #
################################################################################


def _save_to_JSON(path, dictionary):  # rename
    dict_copy = copy.deepcopy(dictionary)
    TruncationsKeys.change_dictionary_keys_from_enum_members_to_strings(dict_copy)
    with open(path, mode='w', encoding='UTF8') as target_file:
        target_file.write(json.dumps(dict_copy))

    return

def save_model_to_JSON(path, dictionary):  # rename
    """ wrapper for _save_to_JSON
    calls verify_model_parameters() before calling _save_to_JSON()
    """
    # verify_model_parameters(dictionary)  modify _verify_func
    # log.debug(f"Saving model to {path:s}")
    _save_to_JSON(path, dictionary)
    return

def _load_from_JSON(path):  # rename
    """returns a dictionary filled with the values stored in the .json file located at path"""

    with open(path, mode='r', encoding='UTF8') as file:
        input_dictionary = json.loads(file.read())

    TruncationsKeys.change_dictionary_keys_from_strings_to_enum_members(input_dictionary)

    for key, value in input_dictionary.items():
        input_dictionary[key] = value
    # add verify
    return input_dictionary

def load_model_from_JSON(path, dictionary=None): # rename
    """
    if kwargs is not provided then returns a dictionary filled with the values stored in the .json file located at path

    if kwargs is provided then all values are overwritten (in place) with the values stored in the .json file located at path
    """

    # no arrays were provided so return newly created arrays after filling them with the appropriate values
    if not bool(dictionary):
        new_model_dict = _load_from_JSON(path)

        # TODO - we might want to make sure that none of the values in the dictionary have all zero values or are None

        # verify_model_parameters(new_model_dict)
        return new_model_dict

    # arrays were provided so fill them with the appropriate values
    # else:
        # verify_model_parameters(dictionary)
        # _load_inplace_from_JSON(path, dictionary)
        # check twice? might as well be cautious for the moment until test cases are written
        # verify_model_parameters(dictionary)

    return
