# system imports
# import re
import pytest
# local imports
from .context import latex_zhz as zhz

from .context import namedtuple_defines as nt


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
        all_z_permutations = [(nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0), nt.general_operator_namedtuple(name='z', rank=0, m=0, n=0)), (nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0), nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0))]
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
        z_term_list = (nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0), nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0))
        function_output = zhz._generate_all_valid_z_connection_permutations(LHS, h, z_term_list, log_invalid=True)
        expected_result = ([((0, 0, 0), (1, 0, 0))], [((0, 0, 0), (0, 0, 0))])
        assert function_output == expected_result

    def test_generate_all_o_h_z_connection_permutations(self):
        LHS = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = zhz.h_operator_namedtuple(rank=0, m=0, n=0)
        valid_z_permutations = [(nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0), nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0))]
        function_output = zhz._generate_all_o_h_z_connection_permutations(LHS, h, valid_z_permutations, found_it_bool=False)
        expected_result = [[None, zhz.disconnected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_h=0, n_h=0, m_l=0, n_l=0)]]
        assert function_output == expected_result

    def test_generate_all_o_h_z_connection_permutations_case_1(self):
        LHS = nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        h = zhz.h_operator_namedtuple(rank=1, m=0, n=1)
        valid_z_permutations = [(nt.general_operator_namedtuple(name=None, rank=0, m=0, n=0), nt.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0))]
        function_output = zhz._generate_all_o_h_z_connection_permutations(LHS, h, valid_z_permutations, found_it_bool=False)
        expected_result = [[None, zhz.connected_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_h=1, n_h=0, m_l=0, n_l=0)]]
        assert function_output == expected_result

    def test_
