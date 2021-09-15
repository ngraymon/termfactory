import os
import sys
from driver import generate_latex_files, generate_python_files

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