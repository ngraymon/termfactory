# system imports
# import re
import pytest

# local imports
from . import context
import latex_zhz as zhz
import namedtuple_defines as nt

# global vars

zero_lhs_op_nt= zhz.connected_lhs_operator_namedtuple(rank=0, m=0, n=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=0)
zero_connected_h_z_op_nt = zhz.connected_h_z_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=0)
zero_disconnected_z_r_nt = zhz.disconnected_z_right_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=0)
zero_disconnected_z_l_nt = zhz.disconnected_z_left_operator_namedtuple(rank=0, m=0, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=0)


class Test_zhz_gen_ops:

    def test_generate_z_operator(self):
        maximum_cc_rank = 1
        function_output = zhz.generate_z_operator(maximum_cc_rank, only_ground_state=False)
        expected_result = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z_1', rank=1, m=0, n=1),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)])
        assert function_output == expected_result


class Test_forming_zhz_latex:

    def test_z_joining_with_z_terms(self):
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = zhz.h_operator_namedtuple(rank=0, m=0, n=0)
        left_z = nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0)
        right_z = nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
        function_output = zhz._z_joining_with_z_terms(LHS, h, left_z, right_z)
        expected_result = False
        assert function_output == expected_result

    def test_generate_valid_z_n_operator_permutations(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h = zhz.h_operator_namedtuple(rank=1, m=0, n=0)
        all_z_permutations = [
            (
                nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
            ),
            (
                nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            )
        ]
        function_output = zhz._generate_valid_z_n_operator_permutations(LHS, h, all_z_permutations)
        expected_result = [
            (
                nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
            )
        ]
        assert function_output == expected_result

    def test_generate_all_valid_z_connection_permutations(self):
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = zhz.h_operator_namedtuple(rank=0, m=0, n=0)
        z_term_list = (
            nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0),
            nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
        )
        function_output = zhz._generate_all_valid_z_connection_permutations(LHS, h, z_term_list, log_invalid=True)
        expected_result = ([((0, 0, 0), (1, 0, 0))], [((0, 0, 0), (0, 0, 0))])
        assert function_output == expected_result

    def test_generate_all_o_h_z_connection_permutations(self):
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = zhz.h_operator_namedtuple(rank=0, m=0, n=0)
        valid_z_permutations = [
            (
                nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            )
        ]
        function_output = zhz._generate_all_o_h_z_connection_permutations(
            LHS,
            h,
            valid_z_permutations,
            found_it_bool=False
        )
        expected_result = [
            [
                None,
                zhz.disconnected_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=1, n_lhs=0,
                    m_h=0, n_h=0,
                    m_l=0, n_l=0)
            ]
        ]
        assert function_output == expected_result

    def test_generate_all_o_h_z_connection_permutations_case_1(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h = zhz.h_operator_namedtuple(rank=1, m=0, n=1)
        valid_z_permutations = [
            (
                nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            )
        ]
        function_output = zhz._generate_all_o_h_z_connection_permutations(
            LHS,
            h,
            valid_z_permutations,
            found_it_bool=False
        )
        expected_result = [
            [
                None,
                zhz.connected_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=0, n_lhs=0,
                    m_h=1, n_h=0,
                    m_l=0, n_l=0
                )
            ]
        ]
        assert function_output == expected_result

    def test_generate_explicit_z_connections_san_1(self):
        '''sanity check 1'''
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h = zhz.h_operator_namedtuple(rank=0, m=0, n=0)
        unique_s_permutations = [
            [
                None,
                zhz.disconnected_z_right_operator_namedtuple(
                    rank=0,
                    m=0, n=0,
                    m_lhs=0, n_lhs=0,
                    m_h=0, n_h=0,
                    m_l=0, n_l=0
                )
            ]
        ]
        function_output = zhz._generate_explicit_z_connections(LHS, h, unique_s_permutations)
        expected_result = [
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [None, zero_disconnected_z_r_nt]
            ]
        ]
        assert function_output == expected_result

    def test_generate_explicit_z_connections_kwargs_check_1(self):
        '''kwargs_check_1: h_kwargs['m_lhs'] != lhs_kwargs['n_h']'''
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = zhz.h_operator_namedtuple(rank=2, m=1, n=1)
        unique_s_permutations = [
            [
                None,
                zhz.connected_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=0, n_lhs=0,
                    m_h=1, n_h=0,
                    m_l=0, n_l=0
                )
            ],
            [
                None,
                zhz.disconnected_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=1, n_lhs=0,
                    m_h=0, n_h=0,
                    m_l=0, n_l=0
                )
            ]
        ]
        function_output = zhz._generate_explicit_z_connections(LHS, h, unique_s_permutations)
        expected_result = [
            [
                zhz.connected_lhs_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_l=0, n_l=0,
                    m_h=0, n_h=1,
                    m_r=0, n_r=0
                ),
                zhz.connected_h_z_operator_namedtuple(
                    rank=2,
                    m=1, n=1,
                    m_lhs=1, n_lhs=0,
                    m_l=0, n_l=0,
                    m_r=0, n_r=1
                ),
                [
                    None,
                    zhz.connected_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                ]
            ]
        ]
        assert function_output == expected_result

    def test_generate_explicit_z_connections_kwargs_check_2(self):
        '''kwargs_check_2: h_kwargs['n_lhs'] != lhs_kwargs['m_h']'''
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = zhz.h_operator_namedtuple(rank=2, m=1, n=1)
        unique_s_permutations = [
            [
                None,
                zhz.connected_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=0, n_lhs=0,
                    m_h=0, n_h=0,
                    m_l=0, n_l=0
                )
            ],
            [
                None,
                zhz.disconnected_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=1, n_lhs=0,
                    m_h=0, n_h=0,
                    m_l=0, n_l=0
                )
            ]
        ]
        function_output = zhz._generate_explicit_z_connections(LHS, h, unique_s_permutations)
        expected_result = []
        assert function_output == expected_result

    def test_filter_out_valid_z_terms(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zhz.h_operator_namedtuple(rank=0, m=0, n=0),
                zhz.h_operator_namedtuple(rank=1, m=0, n=1),
                zhz.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z_left = None
        Z_right = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        total_list = []
        function_output = zhz._filter_out_valid_z_terms(LHS, H, Z_left, Z_right, total_list)
        expected_result = None
        assert function_output == expected_result

    def test_filter_out_valid_z_terms_z_right(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zhz.h_operator_namedtuple(rank=0, m=0, n=0),
                zhz.h_operator_namedtuple(rank=1, m=0, n=1),
                zhz.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z_left = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        Z_right = None
        total_list = []
        function_output = zhz._filter_out_valid_z_terms(LHS, H, Z_left, Z_right, total_list)
        expected_result = None
        assert function_output == expected_result

    def test_filter_out_valid_z_terms_z_left_and_right(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zhz.h_operator_namedtuple(rank=0, m=0, n=0),
                zhz.h_operator_namedtuple(rank=1, m=0, n=1),
                zhz.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z_left = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        Z_right = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        total_list = []
        function_output = zhz._filter_out_valid_z_terms(LHS, H, Z_left, Z_right, total_list)
        expected_result = None
        assert function_output == expected_result


class Test_assign_upper_lower_latex_indices:

    # def test_build_left_z_term(self):
    #     pass, excited state only

    def test_build_hz_term_latex_labels_zero_rank(self):
        """h.rank == 0"""
        h = zero_connected_h_z_op_nt
        offset_dict = {
            'left_upper': 0,
            'summation_index': 0,
            'unlinked_index': 0
        }
        function_output = zhz._build_hz_term_latex_labels(h, offset_dict, color=True)
        expected_result = '\\bh_0'
        assert function_output == expected_result

    def test_build_hz_term_latex_labels_case_1(self):
        """if h.n > 0 """
        h = zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1)
        offset_dict = {
            'left_upper': 0,
            'summation_index': 0,
            'unlinked_index': 0
        }
        function_output = zhz._build_hz_term_latex_labels(h, offset_dict, color=True)
        expected_result = '\\bh^{}_{\\blue{}\\blue{k}\\red{}}'
        assert function_output == expected_result

    def test_build_hz_term_latex_labels_case_2(self):
        """if h.m > 0 """
        h = zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=0)
        offset_dict = {
            'left_upper': 0,
            'summation_index': 0,
            'unlinked_index': 0
        }
        function_output = zhz._build_hz_term_latex_labels(h, offset_dict, color=True)
        expected_result = '\\bh^{\\blue{}\\blue{}\\red{y}}_{}'
        assert function_output == expected_result

    def test_build_right_z_term_rank_zero(self):
        """rank == 0"""
        h = zero_connected_h_z_op_nt
        z_right = zero_disconnected_z_r_nt
        offset_dict = {
            'left_lower': 0,
            'left_upper': 0,
            'h_lower': 0,
            'h_upper': 0,
            'unlinked_index': 0
        }
        function_output = zhz._build_right_z_term(h, z_right, offset_dict, color=True)
        expected_result = '\\bz_0'
        assert function_output == expected_result

    def test_build_right_z_term_case_1(self):
        """z_right.n > 0"""
        h = zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0)
        z_right = zhz.connected_z_right_operator_namedtuple(
            rank=1,
            m=0, n=1,
            m_lhs=0, n_lhs=0,
            m_h=0, n_h=1,
            m_l=0, n_l=0
        )
        offset_dict = {
            'left_lower': 0,
            'left_upper': 0,
            'h_lower': 0,
            'h_upper': 0,
            'unlinked_index': 0
        }
        function_output = zhz._build_right_z_term(h, z_right, offset_dict, color=True)
        expected_result = '\\bz^{}_{\\magenta{}\\blue{k}\\red{}}'
        assert function_output == expected_result

    def test_build_right_z_term_case_2(self):
        """z_right.m > 0"""
        h = zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1)
        z_right = zhz.connected_z_right_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_h=1, n_h=0,
            m_l=0, n_l=0
        )
        offset_dict = {'left_lower': 0, 'left_upper': 0, 'h_lower': 1, 'h_upper': 0, 'unlinked_index': 0}
        function_output = zhz._build_right_z_term(h, z_right, offset_dict, color=True)
        expected_result = '\\bz^{\\magenta{}\\blue{k}\\red{}}_{}'
        assert function_output == expected_result


#################################################################################
# Tests beyond here might rely too much on unreliable inputs from excited states
#################################################################################


class Test_contributions:

    def test_f_zL_h_contributions(self):
        z_left = zero_disconnected_z_l_nt
        h = zero_connected_h_z_op_nt
        function_output = zhz._f_zL_h_contributions(z_left, h)
        expected_result = 0
        assert function_output == expected_result

    def test_f_zL_h_contributions_else(self):
        z_left = zhz.connected_z_left_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_h=1, n_h=0,
            m_r=0, n_r=0
        )
        h = zhz.connected_h_z_operator_namedtuple(
            rank=1,
            m=0, n=1,
            m_lhs=0, n_lhs=0,
            m_l=0, n_l=1,
            m_r=0, n_r=0
        )
        function_output = zhz._f_zL_h_contributions(z_left, h)
        expected_result = 1
        assert function_output == expected_result

    def test_fbar_zL_h_contributions(self):
        z_left = zero_disconnected_z_l_nt
        h = zero_connected_h_z_op_nt
        function_output = zhz._fbar_zL_h_contributions(z_left, h)
        expected_result = 0
        assert function_output == expected_result

    def test_fbar_zL_h_contributions_else(self):
        z_left = zhz.connected_z_left_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_h=1, n_h=0,
            m_r=0, n_r=0
        )
        h = zhz.connected_h_z_operator_namedtuple(
            rank=1,
            m=0, n=1,
            m_lhs=0, n_lhs=0,
            m_l=0, n_l=1,
            m_r=0, n_r=0
        )
        function_output = zhz._fbar_zL_h_contributions(z_left, h)
        expected_result = 0
        assert function_output == expected_result

    def test_f_h_zR_contributions(self):
        h = zero_connected_h_z_op_nt
        z_right = zero_disconnected_z_r_nt
        function_output = zhz._f_h_zR_contributions(h, z_right)
        expected_result = 0
        assert function_output == expected_result

    def test_f_h_zR_contributions_else(self):
        h = zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1)
        z_right = zhz.connected_z_right_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_h=1, n_h=0,
            m_l=0, n_l=0
        )
        function_output = zhz._f_h_zR_contributions(h, z_right)
        expected_result = 0
        assert function_output == expected_result

    def test_fbar_h_zR_contributions(self):
        h = zero_connected_h_z_op_nt
        z_right = zero_disconnected_z_r_nt
        function_output = zhz._fbar_h_zR_contributions(h, z_right)
        expected_result = 0
        assert function_output == expected_result

    def test_fbar_h_zR_contributions_else(self):
        h = zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1)
        z_right = zhz.connected_z_right_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_h=1, n_h=0,
            m_l=0, n_l=0
        )
        function_output = zhz._fbar_h_zR_contributions(h, z_right)
        expected_result = 1
        assert function_output == expected_result

    def test_f_zL_zR_contributions(self):
        z_left = zero_disconnected_z_l_nt
        z_right = zero_disconnected_z_r_nt
        function_output = zhz._f_zL_zR_contributions(z_left, z_right)
        expected_result = 0
        assert function_output == expected_result

    def test_fbar_zL_zR_contributions(self):
        z_left = zero_disconnected_z_l_nt
        z_right = zero_disconnected_z_r_nt
        function_output = zhz._fbar_zL_zR_contributions(z_left, z_right)
        expected_result = 0
        assert function_output == expected_result


class Test_preparing_of_terms:

    def test_prepare_second_z_latex_basic(self):
        term_list = [
            [
                zhz.connected_lhs_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_l=0, n_l=1,
                    m_h=0, n_h=0,
                    m_r=0, n_r=0
                ),
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=1, n_lhs=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=0
                    ),
                    None
                ]
            ],
            [
                zhz.connected_lhs_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_l=0, n_l=0,
                    m_h=0, n_h=1,
                    m_r=0, n_r=0
                ),
                zhz.connected_h_z_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=1, n_lhs=0,
                    m_l=0, n_l=0,
                    m_r=0, n_r=0
                ),
                [
                    zero_disconnected_z_l_nt,
                    None
                ]
            ]
        ]
        function_output = zhz._prepare_second_z_latex(
            term_list,
            split_width=7,
            remove_f_terms=False,
            print_prefactors=False
        )
        expected_result = '(\\bz^{\\blue{}\\red{y}}_{}\\bh_0 + \\bz_0\\bh^{\\blue{}\\blue{}\\red{y}}_{})'
        assert function_output == expected_result

    def test_prepare_second_z_latex_basic_case_1(self):
        """if nof_fs > 0:"""
        term_list = [
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zero_disconnected_z_l_nt,
                    None
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_lhs=0, n_lhs=0,
                    m_l=0, n_l=1,
                    m_r=0, n_r=0
                ),
                [
                    zhz.connected_z_left_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_h=1, n_h=0,
                        m_r=0, n_r=0
                    ),
                    None
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=0, n_lhs=0,
                    m_l=1, n_l=0,
                    m_r=0, n_r=0
                ),
                [
                    zhz.connected_z_left_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                    None
                ]
            ]
        ]
        function_output = zhz._prepare_second_z_latex(
            term_list,
            split_width=7,
            remove_f_terms=False,
            print_prefactors=False
        )
        expected_result = str(
            '(\\bz_0\\bh_0 + f\\bz^{k\\blue{}\\red{}}_{}\\bh^{}_{\\blue{k}\\blue{}\\red{}} + \\bar{f}\\bz^{}_{k\\blue{}'
            '\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{})'
        )
        assert function_output == expected_result

    def test_prepare_third_z_latex(self):
        term_list = [
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    None,
                    zero_disconnected_z_r_nt
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_lhs=0, n_lhs=0,
                    m_l=0, n_l=0,
                    m_r=0, n_r=1
                ),
                [
                    None,
                    zhz.connected_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                ]
            ]
        ]
        function_output = zhz._prepare_third_z_latex(
            term_list,
            split_width=7,
            remove_f_terms=False,
            print_prefactors=False
        )
        expected_result = str(
            '(\\bh_0\\bz_0 + \\bar{f}\\bh^{}_{\\blue{}\\blue{k}\\red{}}\\bz^{\\magenta{}\\blue{k}\\red{}}_{})'
        )
        assert function_output == expected_result

    def test_prepare_third_z_latex_if_case(self):
        term_list = [
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    None,
                    zero_disconnected_z_r_nt
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_lhs=0, n_lhs=0,
                    m_l=0, n_l=0,
                    m_r=0, n_r=1
                ),
                [
                    None,
                    zhz.connected_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=0, n_lhs=0,
                    m_l=0, n_l=0,
                    m_r=1, n_r=0
                ),
                [
                    None,
                    zhz.connected_z_right_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_h=0, n_h=1,
                        m_l=0, n_l=0
                    )
                ]
            ]
        ]
        function_output = zhz._prepare_third_z_latex(
            term_list,
            split_width=7,
            remove_f_terms=False,
            print_prefactors=False
        )
        expected_result = str(
            '(\\bh_0\\bz_0 + \\bar{f}\\bh^{}_{\\blue{}\\blue{k}\\red{}}\\bz^{\\magenta{}\\blue{k}\\red{}}_{} + f\\bh^{'
            '\\blue{}\\blue{k}\\red{}}_{}\\bz^{}_{\\magenta{}\\blue{k}\\red{}})'
        )
        assert function_output == expected_result

    def test_prepare_fourth_z_latex(self):
        term_list = [  # file flag
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zero_disconnected_z_l_nt,
                    zero_disconnected_z_r_nt
                ]
            ],
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0)
                ]
            ],
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1)
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
                [
                    zero_disconnected_z_l_nt,
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0)
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                    zero_disconnected_z_r_nt
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
                [
                    zero_disconnected_z_l_nt,
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0)
                ]
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                    zero_disconnected_z_r_nt
                ]
            ]
        ]
        function_output = zhz._prepare_fourth_z_latex(
            term_list,
            split_width=7,
            remove_f_terms=False,
            print_prefactors=False
        )
        expected_result = str(
            '(\\disconnected{\\bz_0\\bh_0\\bz_0} + \\disconnected{\\bar{f}\\bz^{}_{\\blue{k}\\red{}}\\bh_0\\bz^{\\magen'
            'ta{k}\\blue{}\\red{}}_{}} + \\disconnected{f\\bz^{\\blue{k}\\red{}}_{}\\bh_0\\bz^{}_{\\magenta{k}\\blue{}'
            '\\red{}}} + \\disconnected{\\bar{f}\\bz_0\\bh^{}_{\\blue{}\\blue{k}\\red{}}\\bz^{\\magenta{}\\blue{k}\\red'
            '{}}_{}} + \\disconnected{f\\bz^{k\\blue{}\\red{}}_{}\\bh^{}_{\\blue{k}\\blue{}\\red{}}\\bz_0} + \\disconne'
            'cted{f\\bz_0\\bh^{\\blue{}\\blue{k}\\red{}}_{}\\bz^{}_{\\magenta{}\\blue{k}\\red{}}} + \\disconnected{\\ba'
            'r{f}\\bz^{}_{k\\blue{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{}\\bz_0})'
        )
        assert function_output == expected_result

    def test_long(self):
        term_list = [  # file flag
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zero_disconnected_z_l_nt,
                    zero_disconnected_z_r_nt,
                ],
            ],
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=2),
                    zhz.disconnected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=2, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=1),
                    zhz.disconnected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zero_connected_h_z_op_nt,
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=2, n_r=0),
                    zhz.disconnected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=2),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
                [
                    zero_disconnected_z_l_nt,
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                    zero_disconnected_z_r_nt,
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=1),
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=1),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=1, n_r=0),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=0, n_r=2),
                [
                    zero_disconnected_z_l_nt,
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=2, n_h=0, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=1),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=1),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=1),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=2, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=2, n_h=0, m_r=0, n_r=0),
                    zero_disconnected_z_r_nt,
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=0, n_r=1),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=1, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
                [
                    zero_disconnected_z_l_nt,
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                    zero_disconnected_z_r_nt,
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=0, n_r=1),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=1),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=0),
                [
                    zhz.disconnected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_r=1, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=1, n_r=0),
                    zhz.disconnected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=1, n_r=1),
                [
                    zero_disconnected_z_l_nt,
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=1, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=1),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=1),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=1),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=1, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=1, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=1, m_r=0, n_r=0),
                    zero_disconnected_z_r_nt,
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=0, n_r=1),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=1, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=1, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=0, n_r=1),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=1, m_r=1, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_r=1, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=1),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_r=2, n_r=0),
                [
                    zero_disconnected_z_l_nt,
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=2, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=1, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=2, n_l=0, m_r=0, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=2, m_r=0, n_r=0),
                    zero_disconnected_z_r_nt,
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=1, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=0, n_r=1),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=1, n_l=0),
                ],
            ],
            [
                zero_lhs_op_nt,
                zhz.connected_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_l=1, n_l=0, m_r=1, n_r=0),
                [
                    zhz.connected_z_left_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_r=1, n_r=0),
                    zhz.connected_z_right_operator_namedtuple(rank=2, m=0, n=2, m_lhs=0, n_lhs=0, m_h=0, n_h=1, m_l=0, n_l=1),
                ],
            ],
        ]
        function_output = zhz._prepare_fourth_z_latex(
            term_list,
            split_width=7,
            remove_f_terms=False,
            print_prefactors=False
        )
        expected_result = str(  # file flag
            '(\n\\disconnected{\\bz_0\\bh_0\\bz_0} + \\disconnected{\\bar{f}\\bz^{}_{\\blue{k}\\red{}}\\bh_0\\bz^{\\mag'
            'enta{k}\\blue{}\\red{}}_{}} + \\disconnected{\\bar{f}^{2}\\bz^{}_{\\blue{kl}\\red{}}\\bh_0\\bz^{\\magenta{'
            'kl}\\blue{}\\red{}}_{}} + \\disconnected{f\\bz^{\\blue{k}\\red{}}_{}\\bh_0\\bz^{}_{\\magenta{k}\\blue{}\\r'
            'ed{}}} + \\disconnected{f\\bar{f}\\bz^{\\blue{k}\\red{}}_{\\blue{l}\\red{}}\\bh_0\\bz^{\\magenta{l}\\blue{'
            '}\\red{}}_{\\magenta{k}\\blue{}\\red{}}} + \\disconnected{f^{2}\\bz^{\\blue{kl}\\red{}}_{}\\bh_0\\bz^{}_{'
            '\\magenta{kl}\\blue{}\\red{}}} + \\disconnected{\\bar{f}\\bz_0\\bh^{}_{\\blue{}\\blue{k}\\red{}}\\bz^{\\ma'
            'genta{}\\blue{k}\\red{}}_{}}\n    \\\\  &+  % split long equation\n\\disconnected{\\bar{f}^{2}\\bz^{}_{\\b'
            'lue{k}\\red{}}\\bh^{}_{\\blue{}\\blue{l}\\red{}}\\bz^{\\magenta{k}\\blue{l}\\red{}}_{}} + \\disconnected{f'
            '\\bz^{k\\blue{}\\red{}}_{}\\bh^{}_{\\blue{k}\\blue{}\\red{}}\\bz_0} + \\disconnected{f\\bar{f}\\bz^{\\blue'
            '{k}\\red{}}_{}\\bh^{}_{\\blue{}\\blue{l}\\red{}}\\bz^{\\magenta{}\\blue{l}\\red{}}_{\\magenta{k}\\blue{}\\'
            'red{}}} + \\disconnected{f\\bar{f}\\bz^{k\\blue{}\\red{}}_{\\blue{l}\\red{}}\\bh^{}_{\\blue{k}\\blue{}\\re'
            'd{}}\\bz^{\\magenta{l}\\blue{}\\red{}}_{}} + \\disconnected{f^{2}\\bz^{k\\blue{l}\\red{}}_{}\\bh^{}_{\\blu'
            'e{k}\\blue{}\\red{}}\\bz^{}_{\\magenta{l}\\blue{}\\red{}}} + \\disconnected{\\bar{f}^{2}\\bz_0\\bh^{}_{\\b'
            'lue{}\\blue{kl}\\red{}}\\bz^{\\magenta{}\\blue{kl}\\red{}}_{}} + f\\bar{f}\\bz^{k\\blue{}\\red{}}_{}\\bh^{'
            '}_{\\blue{k}\\blue{l}\\red{}}\\bz^{\\magenta{}\\blue{l}\\red{}}_{}\n    \\\\  &+  % split long equation\nf'
            '\\bar{f}^{2}\\bz^{k\\blue{}\\red{}}_{\\blue{l}\\red{}}\\bh^{}_{\\blue{k}\\blue{m}\\red{}}\\bz^{\\magenta{l'
            '}\\blue{m}\\red{}}_{} + \\disconnected{f^{2}\\bz^{kl\\blue{}\\red{}}_{}\\bh^{}_{\\blue{kl}\\blue{}\\red{}}'
            '\\bz_0} + f^{2}\\bar{f}\\bz^{k\\blue{l}\\red{}}_{}\\bh^{}_{\\blue{k}\\blue{m}\\red{}}\\bz^{\\magenta{}\\bl'
            'ue{m}\\red{}}_{\\magenta{l}\\blue{}\\red{}} + \\disconnected{f\\bz_0\\bh^{\\blue{}\\blue{k}\\red{}}_{}\\bz'
            '^{}_{\\magenta{}\\blue{k}\\red{}}} + \\disconnected{\\bar{f}\\bz^{}_{k\\blue{}\\red{}}\\bh^{\\blue{k}\\blu'
            'e{}\\red{}}_{}\\bz_0} + \\disconnected{f\\bar{f}\\bz^{}_{\\blue{k}\\red{}}\\bh^{\\blue{}\\blue{l}\\red{}}_'
            '{}\\bz^{\\magenta{k}\\blue{}\\red{}}_{\\magenta{}\\blue{l}\\red{}}} + \\disconnected{\\bar{f}^{2}\\bz^{}_{'
            'k\\blue{l}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{}\\bz^{\\magenta{l}\\blue{}\\red{}}_{}}\n    \\\\  &+  '
            '% split long equation\n\\disconnected{f^{2}\\bz^{\\blue{k}\\red{}}_{}\\bh^{\\blue{}\\blue{l}\\red{}}_{}\\b'
            'z^{}_{\\magenta{k}\\blue{l}\\red{}}} + \\disconnected{f\\bar{f}\\bz^{\\blue{k}\\red{}}_{l\\blue{}\\red{}}'
            '\\bh^{\\blue{l}\\blue{}\\red{}}_{}\\bz^{}_{\\magenta{k}\\blue{}\\red{}}} + \\disconnected{f\\bar{f}\\bz_0'
            '\\bh^{\\blue{}\\blue{l}\\red{}}_{\\blue{}\\blue{k}\\red{}}\\bz^{\\magenta{}\\blue{k}\\red{}}_{\\magenta{}'
            '\\blue{l}\\red{}}} + \\bar{f}^{2}\\bz^{}_{k\\blue{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{\\blue{}\\blue'
            '{l}\\red{}}\\bz^{\\magenta{}\\blue{l}\\red{}}_{} + \\bar{f}^{3}\\bz^{}_{k\\blue{l}\\red{}}\\bh^{\\blue{k}'
            '\\blue{}\\red{}}_{\\blue{}\\blue{m}\\red{}}\\bz^{\\magenta{l}\\blue{m}\\red{}}_{} + f^{2}\\bz^{k\\blue{}\\'
            'red{}}_{}\\bh^{\\blue{}\\blue{l}\\red{}}_{\\blue{k}\\blue{}\\red{}}\\bz^{}_{\\magenta{}\\blue{l}\\red{}} +'
            ' \\disconnected{f\\bar{f}\\bz^{k\\blue{}\\red{}}_{l\\blue{}\\red{}}\\bh^{\\blue{l}\\blue{}\\red{}}_{\\blue'
            '{k}\\blue{}\\red{}}\\bz_0}\n    \\\\  &+  % split long equation\nf\\bar{f}^{2}\\bz^{\\blue{k}\\red{}}_{l\\'
            'blue{}\\red{}}\\bh^{\\blue{l}\\blue{}\\red{}}_{\\blue{}\\blue{m}\\red{}}\\bz^{\\magenta{}\\blue{m}\\red{}}'
            '_{\\magenta{k}\\blue{}\\red{}} + f^{2}\\bar{f}\\bz^{k\\blue{}\\red{}}_{\\blue{l}\\red{}}\\bh^{\\blue{}\\bl'
            'ue{m}\\red{}}_{\\blue{k}\\blue{}\\red{}}\\bz^{\\magenta{l}\\blue{}\\red{}}_{\\magenta{}\\blue{m}\\red{}} +'
            ' f^{3}\\bz^{k\\blue{l}\\red{}}_{}\\bh^{\\blue{}\\blue{m}\\red{}}_{\\blue{k}\\blue{}\\red{}}\\bz^{}_{\\mage'
            'nta{l}\\blue{m}\\red{}} + \\disconnected{f^{2}\\bz_0\\bh^{\\blue{}\\blue{kl}\\red{}}_{}\\bz^{}_{\\magenta{'
            '}\\blue{kl}\\red{}}} + f\\bar{f}\\bz^{}_{k\\blue{}\\red{}}\\bh^{\\blue{k}\\blue{l}\\red{}}_{}\\bz^{}_{\\ma'
            'genta{}\\blue{l}\\red{}} + \\disconnected{\\bar{f}^{2}\\bz^{}_{kl\\blue{}\\red{}}\\bh^{\\blue{kl}\\blue{}'
            '\\red{}}_{}\\bz_0} + f\\bar{f}^{2}\\bz^{}_{k\\blue{l}\\red{}}\\bh^{\\blue{k}\\blue{m}\\red{}}_{}\\bz^{\\ma'
            'genta{l}\\blue{}\\red{}}_{\\magenta{}\\blue{m}\\red{}}\n    \\\\  &+  % split long equation\n\\disconnecte'
            'd{f^{2}\\bz_0\\bh^{\\blue{}\\blue{kl}\\red{}}_{}\\bz^{}_{\\magenta{}\\blue{kl}\\red{}}} + f\\bar{f}\\bz^{}'
            '_{k\\blue{}\\red{}}\\bh^{\\blue{k}\\blue{l}\\red{}}_{}\\bz^{}_{\\magenta{}\\blue{l}\\red{}} + \\disconnect'
            'ed{\\bar{f}^{2}\\bz^{}_{kl\\blue{}\\red{}}\\bh^{\\blue{kl}\\blue{}\\red{}}_{}\\bz_0} + f\\bar{f}^{2}\\bz^{'
            '}_{k\\blue{l}\\red{}}\\bh^{\\blue{k}\\blue{m}\\red{}}_{}\\bz^{\\magenta{l}\\blue{}\\red{}}_{\\magenta{}\\b'
            'lue{m}\\red{}} + f^{2}\\bar{f}\\bz^{\\blue{k}\\red{}}_{l\\blue{}\\red{}}\\bh^{\\blue{l}\\blue{m}\\red{}}_{'
            '}\\bz^{}_{\\magenta{k}\\blue{m}\\red{}}\n)'
        )
        assert function_output == expected_result


class Test_build_z_terms:

    def test_build_first_z_term_rank_zero(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        function_output = zhz._build_first_z_term(LHS)
        expected_result = '\\bh_{0,xb}(1-\\delta_{x\\gamma})'
        assert function_output == expected_result

    def test_build_first_z_term_nonzero(self):
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        function_output = zhz._build_first_z_term(LHS)
        expected_result = '\\bh^{i}_{}(1-\\delta_{x\\gamma})'
        assert function_output == expected_result

    def test_build_second_z_term(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zhz.h_operator_namedtuple(rank=0, m=0, n=0),
                zhz.h_operator_namedtuple(rank=1, m=0, n=1),
                zhz.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z_1', rank=1, m=0, n=1),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = zhz._build_second_z_term(LHS, H, Z, remove_f_terms=False)
        expected_result = str(
            '(\\bz_0\\bh_0 + f\\bz^{k\\blue{}\\red{}}_{}\\bh^{}_{\\blue{k}\\blue{}\\red{}} + \\bar{f}\\bz^{}_{k\\blue{}'
            '\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{})'
        )
        assert function_output == expected_result

    def test_build_third_z_term(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zhz.h_operator_namedtuple(rank=0, m=0, n=0),
                zhz.h_operator_namedtuple(rank=1, m=0, n=1),
                zhz.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = zhz._build_third_z_term(LHS, H, Z, remove_f_terms=False)
        expected_result = str(
            '(\\bh_0\\bz_0 + \\bar{f}\\bh^{}_{\\blue{}\\blue{k}\\red{}}\\bz^{\\magenta{}\\blue{k}\\red{}}_{})'
        )
        assert function_output == expected_result

    def test_build_fourth_z_term(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zhz.h_operator_namedtuple(rank=0, m=0, n=0),
                zhz.h_operator_namedtuple(rank=1, m=0, n=1),
                zhz.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z_1', rank=1, m=0, n=1),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = zhz._build_fourth_z_term(LHS, H, Z, remove_f_terms=False)
        expected_result = str(
            '(\\disconnected{\\bz_0\\bh_0\\bz_0} + \\disconnected{\\bar{f}\\bz^{}_{\\blue{k}\\red{}}\\bh_0\\bz^{\\magen'
            'ta{k}\\blue{}\\red{}}_{}} + \\disconnected{f\\bz^{\\blue{k}\\red{}}_{}\\bh_0\\bz^{}_{\\magenta{k}\\blue{}'
            '\\red{}}} + \\disconnected{\\bar{f}\\bz_0\\bh^{}_{\\blue{}\\blue{k}\\red{}}\\bz^{\\magenta{}\\blue{k}\\red'
            '{}}_{}} + \\disconnected{f\\bz^{k\\blue{}\\red{}}_{}\\bh^{}_{\\blue{k}\\blue{}\\red{}}\\bz_0} + \\disconne'
            'cted{f\\bz_0\\bh^{\\blue{}\\blue{k}\\red{}}_{}\\bz^{}_{\\magenta{}\\blue{k}\\red{}}} + \\disconnected{\\ba'
            'r{f}\\bz^{}_{k\\blue{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{}\\bz_0})'
        )
        assert function_output == expected_result

    def test_build_fifth_z_term(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = zhz._build_fifth_z_term(LHS, Z)
        expected_result = '\\dv{\\hat{\\bt}_{0, \\gamma}}{\\tau}\\,\\hat{\\bz}_{0, \\gamma}'
        assert function_output == expected_result

    def test_build_fifth_z_term_else_cases(self):
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = zhz._build_fifth_z_term(LHS, Z)
        expected_result = str(
            '\\dv{\\hat{\\bt}^{i}_{\\gamma}}{\\tau}\\,\\hat{\\bz}_{0, \\gamma} + \\dv{\\hat{\\bt}_{0, \\gamma}}{\\tau}'
            '\\,\\hat{\\bz}^{i}_{\\gamma}'
        )
        assert function_output == expected_result


class Test_gen_latex_eqns:

    def test_generate_z_symmetric_latex_equations(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zhz.h_operator_namedtuple(rank=0, m=0, n=0),
                zhz.h_operator_namedtuple(rank=1, m=0, n=1),
                zhz.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = zhz._generate_z_symmetric_latex_equations(
            LHS,
            H,
            Z,
            only_ground_state=True,
            remove_f_terms=False
        )
        expected_result = str(
            '\\bh_{0,xb}(1-\\delta_{x\\gamma})\\\\&+\\sum\\Big((\\bh_0\\bz_0 + \\bar{f}\\bh_{\\blue{}\\blue{k}\\red{}}'
            '\\bz^{\\magenta{}\\blue{k}\\red{}})\\Big)(1-\\delta_{cb})\\\\&-i\\sum\\Big(\\dv{\\hat{\\bt}_{0, \\gamma}}{'
            '\\tau}\\,\\hat{\\bz}_{0, \\gamma}\\Big)'
        )
        assert function_output == expected_result


class Test_run_main_zhz_for_coverage:

    def test_main_zhz(self):
        truncations = [1, 1, 1, 1]
        zhz.generate_z_t_symmetric_latex(
            truncations,
            only_ground_state=True,
            remove_f_terms=False,
            path="./generated_latex.tex"
        )
