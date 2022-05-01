"""
We take a hands off approach to verifying the correct structure of the truncations.
Only checking at a high level in each individual python file.
For exmaple generate_full_cc_python but non of the otehr functions verify.
There is no intent of warding off malicious/incompetrent users

"""


# system imports
import os
import copy
import json
import itertools as it
from collections import namedtuple

# third party imports

# local imports
from truncation_keys import TruncationsKeys as tkeys

truncation_maximums = {
    'maximum_hamiltonian_rank': 6,  # maximum_h_rank
    'maximum_coupled_cluster_rank': 6,  # maximum_cc_rank
    's_taylor_max_order': 6,
    'omega_max_order': 6,
    'maximum_T_rank': 1,
    'eT_taylor_max_order': 8
}
# depricated
old_trunc_max = {
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
zhz_key_list = [
    'maximum_hamiltonian_rank',
    'maximum_coupled_cluster_rank',
    'omega_max_order'
]

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
    """ Verifies three things about the input dictionary `truncations`.

    First, that the set of keys present in `truncations` is the same set of keys
        present in `key_list`.
    Second that all values are >= 1.
    Third that all values are <= their max values
        as defined in the dictionary `truncation_maximums`.
    """
    for key in key_list:
        assert key in truncations, f"Missing key, {key = :s} not in provided dictionary {truncations=:}"
        assert truncations[key] >= 1, "Truncations need to be positive integers"
        assert truncations[key] <= truncation_maximums[key], f"Key {key} is over the maximum of {truncation_maximums[key]}"


def _verify_zhz_truncations(truncations):
    """ x """

    # verify not empty
    assert bool(truncations), 'Empty dictionary is not correct'

    for key in truncations.keys():
        if key not in tkeys:
            raise Exception(
                f"{key = } is not one of the valid keys for zhz.\n{tkeys.zhz_key_list()}"
            )

    # change in the future possibly? if we use enums for all lists?
    string_copy = truncations.copy()  # this works because all values are immutable
    tkeys.change_dictionary_keys_from_enum_members_to_strings(string_copy)
    __verify_keys(string_copy, zhz_key_list)


def _verify_fcc_truncations(truncations):
    """ x """

    # verify not empty
    assert bool(truncations), 'Empty dictionary is not correct'

    for key in truncations.keys():
        if key not in tkeys:
            raise Exception(
                f"{key = } is not one of the valid keys for fcc.\n{tkeys.fcc_key_list()}"
            )

    # change in the future possibly? if we use enums for all lists?
    string_copy = truncations.copy()  # this works because all values are immutable
    tkeys.change_dictionary_keys_from_enum_members_to_strings(string_copy)
    __verify_keys(string_copy, fcc_key_list)


def _verify_eT_z_t_truncations(truncations):
    """ x """

    # verify not empty
    assert bool(truncations), 'Empty dictionary is not correct'

    for key in truncations.keys():
        if key not in tkeys:
            raise Exception(
                f"{key = } is not one of the valid keys for fcc.\n{tkeys.eT_key_list()}"
            )

    # change in the future possibly? if we use enums for all lists?
    string_copy = truncations.copy()  # this works because all values are immutable
    tkeys.change_dictionary_keys_from_enum_members_to_strings(string_copy)
    __verify_keys(string_copy, eT_z_t_key_list)

################################################################################
#                              Save / Load JSON                                #
################################################################################


def _save_to_JSON(path, dictionary):
    """ x """
    dict_copy = copy.deepcopy(dictionary)
    tkeys.change_dictionary_keys_from_enum_members_to_strings(dict_copy)
    with open(path, mode='w', encoding='UTF8') as target_file:
        target_file.write(json.dumps(dict_copy))
    return


def save_trunc_to_JSON(path, dictionary):
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

    # check that all keys are actually valid
    # before calling `change_dictionary_keys_from_strings_to_enum_members`
    # for key in input_dictionary:
    #     try:
    #         new_key = tkeys(key)
    #         input_dictionary[new_key] = input_dictionary.pop(key)
    #     except ValueError:
    #         print(f'Invalid dictionary {key = }')
    #         print(
    #             'Only these keys allowed:' + ''.join(
    #                 ['\n    ' + str(a) for a in list(tkeys)]
    #             )
    #         )
    #         import sys; sys.exit(0)

    tkeys.change_dictionary_keys_from_strings_to_enum_members(input_dictionary)

    for key, value in input_dictionary.items():
        input_dictionary[key] = value

    if tkeys.key_list_type(input_dictionary) == 'fcc':
        _verify_fcc_truncations(input_dictionary)

    elif tkeys.key_list_type(input_dictionary) == 'eTz':
        _verify_eT_z_t_truncations(input_dictionary)

    else:
        raise Exception("Invalid dictionary")  # TODO flush this out
        # actually redundant bc tkeys.key_list_type would error first

    return input_dictionary


def load_trunc_from_JSON(path, dictionary=None):
    """
    if kwargs is not provided then returns a dictionary filled with the values stored in the .json file located at path
    if kwargs is provided then all values are overwritten (in place) with the values stored in the .json file located at path
    """

    # no arrays were provided so return newly created arrays after filling them with the appropriate values
    if not bool(dictionary):
        new_trunc_dict = _load_from_JSON(path)

        # TODO - we might want to make sure that none of the values in the dictionary have all zero values or are None

        # verify_model_parameters(new_model_dict)
        return new_trunc_dict

    # arrays were provided so fill them with the appropriate values
    # else:
        # verify_model_parameters(dictionary)
        # _load_inplace_from_JSON(path, dictionary)
        # check twice? might as well be cautious for the moment until test cases are written
        # verify_model_parameters(dictionary)

    return
