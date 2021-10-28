# system imports
import sys
import itertools as it

# third party imports

# local import
from driver import generate_latex_files, generate_python_files

# -- python code generators
from code_full_cc import generate_full_cc_python
from code_residual_equations import generate_residual_equations_file
from code_w_equations import generate_w_operator_equations_file
from code_dt_equations import generate_dt_amplitude_equations_file



import log_conf
# for now make a second argument the filepath for logging output
if len(sys.argv) > 1:
    logging_output_filename = str(sys.argv[-1])
    log = log_conf.get_filebased_logger(logging_output_filename)
else:
    log = log_conf.get_stdout_logger()

header_log = log_conf.HeaderAdapter(log, {})
subheader_log = log_conf.SubHeaderAdapter(log, {})


true_or_false = [True, False]


def _generate_all_full_cc_truncations():
    """ x """

    permutations_list = []
    # maximum_h_rank = 2
    # maximum_cc_rank = 2
    # s_taylor_max_order = 2  # this doesn't matter for the Z ansatz
    # omega_max_order = 2

    h, cc, s, omega = 2, 2, 2, 2

    permutations_list.append((h, cc, s, omega))

    return permutations_list


def test_latex_full_cc_ansatz():
    """ x """
    for truncation_permutation in _generate_all_full_cc_truncations():
        for b1, b2, b3 in it.product(true_or_false, repeat=3):
            generate_latex_files(
                truncation_permutation,
                only_ground_state=b1,
                remove_f_terms=b2,
                thermal=b3,
                file='full cc'
            )


def _generate_all_eT_truncations():
    """ x """

    permutations_list = []
    # maximum_h_rank = 2
    # maximum_cc_rank = 2
    # maximum_T_rank = 2
    # eT_taylor_max_order = 2
    # omega_max_order = 2

    h, cc, T, eT, omega = 2, 2, 2, 2, 2

    permutations_list.append((h, cc, T, eT, omega))

    return permutations_list


def test_latex_eT_z_t_ansatz():
    """ x """
    for truncation_permutation in _generate_all_eT_truncations():
        for b1, b2, b3 in it.product(true_or_false, repeat=3):

            # temporary fix as
            # ```The excited state TZT terms for the 5th ansatz has not been properly implemented```
            # per line 1610 in `latex_eT_zhz.py`, will hang due to many asserts that are working as intended
            if b1 is False:
                continue

            generate_latex_files(
                truncation_permutation,
                only_ground_state=b1,
                remove_f_terms=b2,
                thermal=b3,
                file='eT_z_t ansatz'
            )

def test_latex_w_equations():
    """ x """
    for truncation_permutation in _generate_all_full_cc_truncations():#think full_cc should cover most unless we wanna expand the max_h rank
        for b2, b3 in it.product(true_or_false, repeat=2):
            generate_latex_files(
                truncation_permutation,
                only_ground_state=True, # the excited_state_w_equations_latex has not been verified, generate_t_terms_group all_combinations_list itertools product hangs on False
                remove_f_terms=b2,
                thermal=b3,
                file='w equations'
            )

def test_latex_z_t_ansatz():
    """ x """
    for truncation_permutation in _generate_all_full_cc_truncations():#think full_cc should cover most unless we wanna expand the max_h rank
        for b1, b2, b3 in it.product(true_or_false, repeat=3):
            generate_latex_files(
                truncation_permutation,
                only_ground_state=b1, #The excited state ZT terms for the 5th ansatz has not been properly implemented
                remove_f_terms=b2,
                thermal=b3,
                file='z_t ansatz'
            )

def test_generate_latex_files():
    """ x """
    test_latex_full_cc_ansatz()
    test_latex_eT_z_t_ansatz()
    test_latex_w_equations()
    test_latex_z_t_ansatz()


def test_code_full_cc_ansatz():
    """ x """
    for truncation_permutation in _generate_all_full_cc_truncations():
        for b1, b2 in it.product(true_or_false, repeat=2):
            generate_python_files(
                truncation_permutation,
                only_ground_state=b1,
                thermal=b2
            )


def test_code_residual_equations(MAX_R=6, MAX_H=2):
    """ MAX_H can't be higher than 2 because of the `h_dict` in `code_residual_equations.py`
    This can be fixed by making the dictionary larger/factoring it out/generating it on the fly...
    maybe there is a class???
    """
    for max_residual_order in range(0, MAX_R+1):
        for maximum_h_rank in range(0, MAX_H+1):
            generate_residual_equations_file(max_residual_order, maximum_h_rank)


def test_code_w_operator(MAX_W=6):
    """ x """
    for max_w_order in range(0, MAX_W+1):
        generate_w_operator_equations_file(max_w_order)


def test_code_dt_amplitude(MAX_dt=6):
    """ x """
    for dt_order in range(0, MAX_dt+1):
        generate_dt_amplitude_equations_file(dt_order)


def test_generate_python_files():
    """ x """
    test_code_full_cc_ansatz()
    test_code_residual_equations()
    test_code_w_operator()
    test_code_dt_amplitude()


#  begin testing portion below is a test for CI
if (__name__ == '__main__'):
    test_generate_latex_files()
    test_generate_python_files()
