# system imports

# third party imports

# local imports
from truncations import _verify_eT_z_t_truncations, _verify_fcc_truncations
from truncation_keys import TruncationsKeys as tkeys

# -- latex code generators
from latex_full_cc import generate_full_cc_latex
from latex_w_equations import ground_state_w_equations_latex, excited_state_w_equations_latex
from latex_zhz import generate_z_t_symmetric_latex
from latex_eT_zhz import generate_eT_z_t_symmetric_latex

# -- python code generators
from code_eT_zhz import generate_eT_zhz_python
from code_full_cc import generate_full_cc_python
from code_residual_equations import generate_residual_equations_file
from code_w_equations import generate_w_operator_equations_file
from code_dt_equations import generate_dt_amplitude_equations_file


# def generate_latex_files(truncations, only_ground_state=True, remove_f_terms=False, thermal=False, file=None):
#     """ only generates .tex files to be compiled into pdf files """

#     if file == 'full cc':
#         if only_ground_state:
#             generate_full_cc_latex(truncations, only_ground_state=True, path="./ground_state_full_cc_equations.tex")
#         else:
#             generate_full_cc_latex(truncations, only_ground_state=False, path="./full_cc_equations.tex")

#     # this doesn't care about the truncation numbers
#     elif file == 'w equations':

#         """
#         eventually we want to merge both the
#         `ground_state_w_equations_latex`
#         and the
#         `excited_state_w_equations_latex`
#         functions
#         """

#         max_w_order = 5  # this is the

#         if only_ground_state:
#             path = "./ground_state_w_equations.tex"
#             ground_state_w_equations_latex(max_w_order, path)
#         else:
#             path = "./excited_state_w_equations.tex"
#             # assert False, 'the excited_state_w_equations_latex has not been verified'
#             print('WARNING: The excited_state_w_equations_latex has not been verified')
#             excited_state_w_equations_latex(max_w_order, path)

#     # the `s_taylor_max_order` isn't relevant for this execution pathway
#     elif file == 'z_t ansatz':
#         f_term_string = "_no_f_terms" if remove_f_terms else ''

#         if only_ground_state:
#             path = f"./ground_state_z_t_symmetric_equations{f_term_string}.tex"
#         else:
#             path = f"./z_t_symmetric_equations{f_term_string}.tex"

#         generate_z_t_symmetric_latex(truncations, only_ground_state, remove_f_terms, path)

#     # the `s_taylor_max_order` isn't relevant for this execution pathway
#     elif file == 'eT_z_t ansatz':
#         f_term_string = "_no_f_terms" if remove_f_terms else ''

#         if only_ground_state:
#             path = f"./ground_state_eT_z_t_symmetric_equations{f_term_string}.tex"
#         else:
#             path = f"./eT_z_t_symmetric_equations{f_term_string}.tex"

#         generate_eT_z_t_symmetric_latex(truncations, only_ground_state, remove_f_terms, path)

#     else:
#         raise Exception(f"Wrong file type specified in {file=}")

#     return


def generate_python_files(truncations, only_ground_state=True, thermal=False):
    """ generates .py files which will be used when calculating desired quantities """

    generate_full_cc_python(truncations, only_ground_state)

    generate_eT_zhz_python(truncations, only_ground_state)

    # max_residual_order = 6
    # generate_residual_equations_file(max_residual_order, truncations[0])
    # max_w_order = 6
    # generate_w_operator_equations_file(max_w_order)
    # dt_order = 6
    # generate_dt_amplitude_equations_file(dt_order)
    return


# def main():
    """ x """

# fcc_trunc = {
#     tkeys.H: 2,
#     tkeys.CC: 6,
#     tkeys.S: 2,
#     tkeys.P: 3
# }

# _verify_fcc_truncations(fcc_trunc)

# # for the 'eT_z_t ansatz' only
# eT_trunc = {
#     tkeys.H: 2,
#     tkeys.CC: 4,
#     tkeys.T: 1,
#     tkeys.eT: 4,
#     tkeys.P: 4
# }

# _verify_eT_z_t_truncations(eT_trunc)
def _make_trunc(tuple):
    # temp, makes a fcc ENUM
    trunc={
    tkeys.H: tuple[0],
    tkeys.CC: tuple[1],
    tkeys.S: tuple[2],
    tkeys.P: tuple[3]
    }
    return trunc



def _generate_latex(trunc, **kwargs):
    default_kwargs = {
        'only_ground_state': True,
        'remove_f_terms': False,
        'ansatz': 'full cc'
    }
    # if empty dict
    if not bool(kwargs):
        kwargs = default_kwargs
    else:
        default_kwargs.update(kwargs)

    if kwargs['ansatz'] == 'full cc':
        assert tkeys.key_list_type(trunc) == 'fcc', "Truncations must be fcc type"
        default_kwargs.pop('ansatz')
        default_kwargs.pop('remove_f_terms')
        _gen_wrapper_full_cc_latex(trunc, **default_kwargs)

    elif kwargs['ansatz'] == 'z_t ansat':
        assert tkeys.key_list_type(trunc) == 'fcc', "Truncations must be fcc type"
        default_kwargs.pop('ansatz')
        _gen_wrapper_z_t_latex(trunc, **default_kwargs)

    elif kwargs['ansatz'] == 'eT_z_t ansatz':
        assert tkeys.key_list_type(trunc) == 'eTz', "Truncations must be eTz type"
        default_kwargs.pop('ansatz')
        _gen_wrapper_eT_z_t_latex(trunc, **default_kwargs)

    else:
        string = (
            f'Invalid {kwargs["ansatz"] = }\n'
            "Only 'full cc' or 'z_t ansatz' or 'eT_z_t ansatz' are valid values.\n"
        )
        raise Exception(string)

    # file='eT_z_t ansatz'


def _gen_wrapper_full_cc_latex(truncations, **kwargs):

    f_term_string = "_no_f_terms" if kwargs['remove_f_terms'] else ''
    gs_string = "ground_state_" if kwargs['only_ground_state'] else ''

    path = f"./{gs_string}full_cc_equations{f_term_string}.tex"

    generate_full_cc_latex(truncations, **kwargs)


def _gen_wrapper_z_t_latex(truncations, **kwargs):
    # the 's_taylor_max_order' isn't releveant for this execution pathway

    f_term_string = "_no_f_terms" if kwargs['remove_f_terms'] else ''
    gs_string = "ground_state_" if kwargs['only_ground_state'] else ''

    path = f"./{gs_string}z_t_symmetric_equations{f_term_string}.tex"

    generate_z_t_symmetric_latex(truncations, **kwargs)


def _gen_wrapper_eT_z_t_latex(truncations, **kwargs):
    # the 's_taylor_max_order' isn't releveant for this execution pathway

    f_term_string = "_no_f_terms" if kwargs['remove_f_terms'] else ''
    gs_string = "ground_state_" if kwargs['only_ground_state'] else ''

    path = f"./{gs_string}eT_z_t_symmetric_equations{f_term_string}.tex"

    generate_eT_z_t_symmetric_latex(truncations, **kwargs)

# def funcs():
#     pass
#         generate_latex_files(eT_z_t_truncations,only_ground_state=True,remove_f_terms=False,thermal=False,file='eT_z_t ansatz')
#         generate_latex_files(truncations,only_ground_state=False,remove_f_terms=False,thermal=False,file='z_t ansatz')
#         generate_latex_files(truncations,only_ground_state=True,remove_f_terms=False,thermal=False,file='full cc')

#         generate_eT_zhz_python(eT_z_t_truncations, only_ground_state=True)
#         generate_full_cc_python(truncations, only_ground_state=True)

# _generate_latex(eT_trunc, ansatz='eT_z_t ansatz', only_ground_state=True, remove_f_terms=True)

# print("We reached the end of main")
