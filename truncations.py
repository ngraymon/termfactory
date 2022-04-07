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

""" contains the Truncations Keys """

class TruncationsKeys(Enum):
    """The TruncationsKeys, which are the keys (strings) used in the .json files to identify the corresponding values
    """

    max_h_rank = 'maximum_hamiltonian_rank'
    max_cc_rank = 'maximum_coupled_cluster_rank'
    max_s_order = 's_taylor_max_order'
    max_proj = 'omega_max_order'
    max_T_rank = 'maximum_T_rank'
    max_eT_order = 'eT_taylor_max_order'

    # aliases for the enum members
    H = max_h_rank
    CC = max_cc_rank
    S = max_s_order
    P = max_proj
    T = max_T_rank
    eT = max_eT_order

    @classmethod
    def change_dictionary_keys_from_enum_members_to_strings(cls, input_dict):
        """ does what it says """
        for key, value in list(input_dict.items()):
            if key in cls:
                input_dict[key.value] = value
                del input_dict[key]
        return

    @classmethod
    def change_dictionary_keys_from_strings_to_enum_members(cls, input_dict):
        """ does what it says """
        for key, value in list(input_dict.items()):
            input_dict[cls(key)] = value
            del input_dict[key]
        return

    @classmethod
    def fcc_key_list(cls):
        """ returns a list of all enum members that are omitted from the .json file if all of their array's values are 0
        """
        #  maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order = truncations
        return [cls.H, cls.CC, cls.S, cls.P]

    @classmethod
    def eT_key_list(cls):
        """ returns a list of all enum members that are omitted from the .json file if all of their array's values are 0
        """
        #  maximum_h_rank, maximum_cc_rank, maximum_T_rank, eT_taylor_max_order, omega_max_order = truncations
        return [cls.H, cls.CC, cls.T, cls.eT, cls.P]

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

################################################################################
#                              Save / Load JSON                                #
################################################################################


def _save_to_JSON(path, dictionary):
    dict_copy = copy.deepcopy(dictionary)
    TruncationsKeys.change_dictionary_keys_from_enum_members_to_strings(dict_copy)
    with open(path, mode='w', encoding='UTF8') as target_file:
        target_file.write(json.dumps(dict_copy))

    return

def save_model_to_JSON(path, dictionary):
    """ wrapper for _save_to_JSON
    calls verify_model_parameters() before calling _save_to_JSON()
    """
    # verify_model_parameters(dictionary)  modify _verify_func
    # log.debug(f"Saving model to {path:s}")
    _save_to_JSON(path, dictionary)
    return

def _load_from_JSON(path):
    """returns a dictionary filled with the values stored in the .json file located at path"""

    with open(path, mode='r', encoding='UTF8') as file:
        input_dictionary = json.loads(file.read())

    TruncationsKeys.change_dictionary_keys_from_strings_to_enum_members(input_dictionary)

    for key, value in input_dictionary.items():
        input_dictionary[key] = value
    # add verify
    return input_dictionary
