# system imports
import os
import sys

# third party imports

# local import
from driver import generate_latex_files, generate_python_files

import log_conf

# for now make a second argument the filepath for logging output
if len(sys.argv) > 1:
    logging_output_filename = str(sys.argv[1])
    log = log_conf.get_filebased_logger(logging_output_filename)
else:
    log = log_conf.get_stdout_logger()

header_log = log_conf.HeaderAdapter(log, {})
subheader_log = log_conf.SubHeaderAdapter(log, {})


#  begin testing portion below is a test for CI
if (__name__ == '__main__'):

    maximum_h_rank = 2
    maximum_cc_rank = 2
    s_taylor_max_order = 2  # this doesn't matter for the Z ansatz
    omega_max_order = 2

    # for the 'z_t ansatz'
    truncations = maximum_h_rank, maximum_cc_rank, s_taylor_max_order, omega_max_order

    # for the 'eT_z_t ansatz' only
    maximum_h_rank = 2
    maximum_cc_rank = 2
    maximum_T_rank = 2
    eT_taylor_max_order = 2
    omega_max_order = 2

    # need to have truncation of e^T
    eT_z_t_truncations = maximum_h_rank, maximum_cc_rank, maximum_T_rank, eT_taylor_max_order, omega_max_order

    generate_latex_files(
        eT_z_t_truncations,
        only_ground_state=True,
        remove_f_terms=False,
        thermal=False,
        file='eT_z_t ansatz'
    )
    generate_python_files(truncations, only_ground_state=True, thermal=False)
