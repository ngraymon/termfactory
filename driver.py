# system imports
import os
import sys
import argparse

# third party imports

# local imports

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


def generate_latex_files(truncations, only_ground_state=True, remove_f_terms=False, thermal=False, file=None):
    """ only generates .tex files to be compiled into pdf files """

    if file == 'full cc':
        if only_ground_state:
            generate_full_cc_latex(truncations, only_ground_state=True, path="./ground_state_full_cc_equations.tex")
        else:
            generate_full_cc_latex(truncations, only_ground_state=False, path="./full_cc_equations.tex")

    # this doesn't care about the truncation numbers
    elif file == 'w equations':

        """
        eventually we want to merge both the
        `ground_state_w_equations_latex`
        and the
        `excited_state_w_equations_latex`
        functions
        """

        max_w_order = 5  # this is the

        if only_ground_state:
            path = "./ground_state_w_equations.tex"
            ground_state_w_equations_latex(max_w_order, path)
        else:
            path = "./excited_state_w_equations.tex"
            # assert False, 'the excited_state_w_equations_latex has not been verified'
            print('WARNING: The excited_state_w_equations_latex has not been verified')
            excited_state_w_equations_latex(max_w_order, path)

    # the `s_taylor_max_order` isn't relevant for this execution pathway
    elif file == 'z_t ansatz':
        f_term_string = "_no_f_terms" if remove_f_terms else ''

        if only_ground_state:
            path = f"./ground_state_z_t_symmetric_equations{f_term_string}.tex"
        else:
            path = f"./z_t_symmetric_equations{f_term_string}.tex"

        generate_z_t_symmetric_latex(truncations, only_ground_state, remove_f_terms, path)

    # the `s_taylor_max_order` isn't relevant for this execution pathway
    elif file == 'eT_z_t ansatz':
        f_term_string = "_no_f_terms" if remove_f_terms else ''

        if only_ground_state:
            path = f"./ground_state_eT_z_t_symmetric_equations{f_term_string}.tex"
        else:
            path = f"./eT_z_t_symmetric_equations{f_term_string}.tex"

        generate_eT_z_t_symmetric_latex(truncations, only_ground_state, remove_f_terms, path)

    else:
        raise Exception(f"Wrong file type specified in {file=}")

    return


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


def dump_all_stdout_to_devnull():
    sys.stdout = open(os.devnull, 'w')


def prepare_parsed_arguments():
    """ Wrapper for argparser setup """

    # formatclass = argparse.RawDescriptionHelpFormatter
    # formatclass = argparse.RawTextHelpFormatter
    formatclass = argparse.ArgumentDefaultsHelpFormatter  # I liked this the best
    # formatclass = argparse.MetavarTypeHelpFormatter

    # parse the arguments
    parser = argparse.ArgumentParser(description="Code/Latex Generator", formatter_class=formatclass)
    parser.add_argument('-l', '--log_path', type=str, metavar='/path/filename.txt', help='path to log file')

    return parser.parse_args()


if (__name__ == '__main__'):
    import log_conf

    pargs = prepare_parsed_arguments()

    if pargs.log_path is None:
        log = log_conf.get_stdout_logger()
    else:
        logging_output_filename = str(pargs.log_path)
        log = log_conf.get_filebased_logger(logging_output_filename)

    header_log = log_conf.HeaderAdapter(log, {})
    subheader_log = log_conf.SubHeaderAdapter(log, {})

    # dump_all_stdout_to_devnull()   # calling this removes all prints / logs from stdout
    # log.setLevel('CRITICAL')

    maximum_h_rank = 2
    maximum_cc_rank = 6
    s_taylor_max_order = 2  # this doesn't matter for the Z ansatz
    omega_max_order = 4

    # for the 'z_t ansatz'
    truncations = maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order

    # for the 'eT_z_t ansatz' only
    maximum_h_rank = 2
    maximum_cc_rank = 4
    maximum_T_rank = 1
    eT_taylor_max_order = 4
    omega_max_order = 4

    # need to have truncation of e^T
    eT_z_t_truncations = maximum_h_rank, maximum_cc_rank, maximum_T_rank, eT_taylor_max_order, omega_max_order

    switch = 1

    if switch == 1:
        generate_latex_files(
            eT_z_t_truncations,
            only_ground_state=True,
            remove_f_terms=False,
            thermal=False,
            file='eT_z_t ansatz'
        )
    if switch == 2:
        generate_latex_files(
            truncations,
            only_ground_state=False,
            remove_f_terms=False,
            thermal=False,
            file='z_t ansatz'
        )
    if switch == 3:
        generate_latex_files(
            truncations,
            only_ground_state=True,
            remove_f_terms=False,
            thermal=False,
            file='full cc'
        )
    if switch == 4:
        generate_eT_zhz_python(eT_z_t_truncations, only_ground_state=True)
        # generate_full_cc_python(truncations, only_ground_state=True)

    print("We reached the end of main")
