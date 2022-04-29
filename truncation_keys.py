""" contains the Truncations Keys """

from enum import Enum


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

    @classmethod
    def key_list_type(cls, dictionary):
        """ Returns a short string indicating what type of truncation dictionary etc. """
        if len(dictionary) == len(cls.fcc_key_list()):
            return 'fcc'
        elif len(dictionary) == len(cls.eT_key_list()):
            return 'eTz'
        else:
            raise Exception(f'Invalid dictionary?!\n{dictionary = }\n')
