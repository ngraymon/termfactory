# system imports
# import re
import pytest
# local imports
from .context import latex_full_cc as fcc

from .context import namedtuple_defines as nt

# TODO add functions where multiple tests are encased in one function for future debugging
# TODO fix long lists
class Test_Operator_Gens:

    def test_generate_omega_operator(self):
        function_output = fcc.generate_omega_operator(maximum_cc_rank=2, omega_max_order=3)
        expected_result = nt.omega_namedtuple(
            maximum_rank=2,
            operator_list=[
                nt.general_operator_namedtuple(name='', rank=0, m=0, n=0),
                nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1),
                nt.general_operator_namedtuple(name='d', rank=1, m=1, n=0),
                nt.general_operator_namedtuple(name='bb', rank=2, m=0, n=2),
                nt.general_operator_namedtuple(name='db', rank=2, m=1, n=1),
                nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
            ]
        )
        assert function_output == expected_result

    def test_generate_full_cc_hamiltonian_operator(self):
        function_output = fcc.generate_full_cc_hamiltonian_operator(maximum_rank=2)
        expected_result = nt.hamiltonian_namedtuple(
            maximum_rank=2,
            operator_list=[
                fcc.h_operator_namedtuple(rank=0, m=0, n=0),
                fcc.h_operator_namedtuple(rank=1, m=0, n=1),
                fcc.h_operator_namedtuple(rank=2, m=0, n=2),
                fcc.h_operator_namedtuple(rank=1, m=1, n=0),
                fcc.h_operator_namedtuple(rank=2, m=1, n=1),
                fcc.h_operator_namedtuple(rank=2, m=2, n=0)
            ]
        )
        assert function_output == expected_result

    def test_generate_s_operator(self):
        function_output = fcc.generate_s_operator(maximum_cc_rank=2, only_ground_state=False)
        expected_result = fcc.s_operator_namedtuple(
            maximum_rank=2,
            operator_list=[
                nt.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                nt.general_operator_namedtuple(name='s^1', rank=1, m=1, n=0),
                nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1),
                nt.general_operator_namedtuple(name='s^2', rank=2, m=2, n=0)])
        assert function_output == expected_result

    def test_generate_s_taylor_expansion(self):
        # kinda messy produces all combinations with lists of lists, gonna come back to this maybe not worth testing
        fcc.generate_s_taylor_expansion(maximum_cc_rank=3, s_taylor_max_order=1, only_ground_state=False)


@pytest.fixture
def lots_of_omegas(rank=2, truncate=3):
    return fcc.generate_omega_operator(maximum_cc_rank=rank, omega_max_order=truncate)


@pytest.fixture
def omega_zero():
    return nt.general_operator_namedtuple(name='', rank=0, m=0, n=0)


@pytest.fixture
def omega_db():
    return nt.general_operator_namedtuple(name='db', rank=2, m=1, n=1)


@pytest.fixture
def h_0():
    return fcc.h_operator_namedtuple(rank=0, m=0, n=0)


@pytest.fixture
def s_term_identity():
    return nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)


class Test_Validate_Op_Pairings:

    def test_t_joining_with_t_terms(self, omega_zero, h_0, s_term_identity):
        """ x """

        args = [
            omega_zero,
            h_0,
            [(s_term_identity), ],
        ]

        assert fcc._t_joining_with_t_terms(*args, nof_creation_ops=1) is False

        # change s term
        args[2] = [nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1), ]
        assert fcc._t_joining_with_t_terms(*args, nof_creation_ops=1) is True

    def test_omega_joining_with_itself(self, omega_db, omega_zero, h_0, s_term_identity):
        """ x """

        """ (omega.m == 0) or (omega.n == 0) return False """
        args = (
            omega_zero,
            h_0,
            [s_term_identity, ],
        )
        assert fcc._omega_joining_with_itself(*args) is False

        """ (omega.n > 0 and h.m > 0) or (omega.m > 0 and h.n > 0) return False """
        args = (
            omega_db,
            fcc.h_operator_namedtuple(rank=2, m=1, n=1),
            [s_term_identity, ],
        )
        assert fcc._omega_joining_with_itself(*args) is False

        """ for s in s_list: if (omega.n > 0 and s.m > 0) or (omega.m > 0 and s.n > 0): return False """
        args = (
            omega_db,
            h_0,
            [nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1), ],
        )
        assert fcc._omega_joining_with_itself(*args) is False

        """ else == True """
        args = (
            omega_db,
            h_0,
            [s_term_identity, ],
        )
        assert fcc._omega_joining_with_itself(*args) is True

    def test_h_joining_with_itself(self, omega_zero, h_0, s_term_identity):
        # (h.m == 0) or (h.n == 0): return False
        omega = omega_zero
        h = h_0
        s_list = [s_term_identity, ]
        function_output = fcc._h_joining_with_itself(omega, h, s_list)
        assert function_output is False

        # if (omega.n > 0 and h.m > 0) or (omega.m > 0 and h.n > 0): return False
        omega = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list = [nt.general_operator_namedtuple(name='s^1', rank=1, m=1, n=0)]
        function_output = fcc._h_joining_with_itself(omega, h, s_list)
        assert function_output is False

        # for s in s_list: if (omega.n > 0 and s.m > 0) or (omega.m > 0 and s.n > 0): return False
        omega = omega_zero
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list = [nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1)]
        function_output = fcc._h_joining_with_itself(omega, h, s_list)
        assert function_output is False

        # else == True
        omega = omega_zero
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list = [s_term_identity, ]
        function_output = fcc._h_joining_with_itself(omega, h, s_list)
        assert function_output is True


class Test_Gen_Ops:

    def test_generate_valid_s_n_operator_permutations(self, omega_zero, h_0, s_term_identity):
        omega = omega_zero
        h = h_0
        s_series_term = [[s_term_identity, ], ]
        function_output = fcc._generate_valid_s_n_operator_permutations(omega, h, s_series_term)
        expected_result = [[s_term_identity, ], ]
        assert expected_result == function_output

    def test_generate_all_valid_t_connection_permutations(self):
        omega_2_0 = nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
        h_1_1 = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_term_list = [
            nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
            nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1),
        ]
        function_output = fcc._generate_all_valid_t_connection_permutations(
            omega_2_0, h_1_1, s_term_list, log_invalid=True
        )

        expected_result = ([((0, 0), (0, 1))], [((1, 1), (1, 0)), ((2, 0), (0, 1))])

        assert expected_result == function_output

    def test_generate_all_omega_h_connection_permutations(self, omega_zero, h_0, s_term_identity):
        omega = nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
        h = fcc.h_operator_namedtuple(rank=2, m=2, n=0)
        valid_permutations = [[nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2), nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)]]
        function_output = fcc._generate_all_omega_h_connection_permutations(omega, h, valid_permutations, found_it_bool=False)
        expected_result = [
            [
                fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2)
            ],
            [
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),
                fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1)
            ],
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0)
            ]
        ]
        assert expected_result == function_output

    def test_remove_duplicate_s_permutations(self):  # TODO add dupe test
        s_list = [[fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)], [fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)]]
        function_output = fcc._remove_duplicate_s_permutations(s_list)
        expected_result = {(fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)}
        assert expected_result == function_output

    def test_generate_explicit_connections(self, omega_zero, h_0, s_term_identity):

        # normal input
        omega = omega_zero
        h = fcc.h_operator_namedtuple(rank=2, m=0, n=2)
        unique_s_permutations = {(fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0))}
        function_output = fcc._generate_explicit_connections(omega, h, unique_s_permutations)
        expected_result = [
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0, 0], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
                )
            ]
        ]
        assert expected_result == function_output

    # def test_generate_explicit_connections_elif_case(self):
    #     # elif h_kwargs['n_o'] != o_kwargs['m_h']
    #     omega = 2
    #     h = 
    

        # TODO if req, also add raise excepts for if states?
        # invalid term case 1
        # invalid term case 2

    def test_remove_f_zero_terms(self):
        # TODO exclude log replace with raise exceptions
        labeled_permutations = [
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0, 0], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
                )
            ]
        ]
        function_output = fcc._remove_f_zero_terms(labeled_permutations)
        assert function_output == labeled_permutations

    def test_filter_out_valid_s_terms(self, omega_zero, h_0, s_term_identity):
        # TODO test list mutation
        omega = omega_zero
        H = fcc.generate_full_cc_hamiltonian_operator(maximum_rank=2)

        s_series_term = nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)
        term_list = []
        total_list = []
        function_output = fcc._filter_out_valid_s_terms(omega, H, s_series_term, term_list, total_list, remove_f_terms=True)
        assert function_output is None

    def test_seperate_s_terms_by_connection(self):
        # TODO add further tests
        total_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
                (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),)
            ]
        ]
        function_output = fcc._seperate_s_terms_by_connection(total_list)
        expected_result = (
            [
                [
                    fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                    fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                    (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)
                ],
                [
                    fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                    fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
                    (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)
                ],
                [
                    fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                    fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]),
                    (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),)
                ]
            ], [], [])
        assert expected_result == function_output


class Test_upper_lower_latex_indicies:

    def test_build_h_term_latex_labels_normal(self):
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        function_output = fcc._build_h_term_latex_labels(h, condense_offset=0, color=True)
        expected_result = '\\bh^' + r'{\blue{}\red{}}_'r'{\blue{i}\red{}}'
        assert function_output == expected_result

    def test_build_h_term_latex_labels_zero_case(self):
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        function_output = fcc._build_h_term_latex_labels(h, condense_offset=0, color=True)
        expected_result = '\\bh_0'
        assert function_output == expected_result

    def test_build_t_term_latex_labels_zero(self):
        term = fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}
        function_output = fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        expected_result = '^{}_{}'
        assert function_output == expected_result

    def test_build_t_term_latex_labels_subscript(self):
        term = fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 1, 'summation_lower': 0, 'unlinked': 0}
        function_output = fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        expected_result = '^{}_' + r'{\blue{i}\red{}}'
        assert function_output == expected_result

    def test_build_t_term_latex_labels_superscript(self):
        term = fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}
        function_output = fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        expected_result = r'^{\blue{i}\red{}}_{}'
        assert function_output == expected_result

    def test_build_t_term_latex(self):
        s = fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
        function_output = fcc._build_t_term_latex(s, h=None)
        expected_result = r'\bt^{\blue{}\red{z}}_{}'
        assert function_output == expected_result

    def test_build_t_term_latex_non_none_h(self):  # TODO is this really testing the function?
        s = fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        function_output = fcc._build_t_term_latex(s, h)
        expected_result = r'\bt^{\blue{}\red{z}}_{}'
        assert function_output == expected_result


class Test_latex_writing:

    def test_validate_s_terms_valid(self):
        s_list = (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
        function_output = fcc._validate_s_terms(s_list)
        expected_result = None
        assert function_output == expected_result

    def test_validate_s_terms_invalid(self):
        s_list = ("weeee")
        with pytest.raises(AssertionError):
            fcc._validate_s_terms(s_list)

    def test_generate_linked_common_terms(self):
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
            ]
        ]
        function_output = fcc._generate_linked_common_terms(term_list)
        expected_result = [[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)]]
        assert function_output == expected_result

    def test_prepare_condensed_terms(self):
        # link condesed
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0], n_t=[1]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=1, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),)
            ]
        ]
        function_output = fcc.prepare_condensed_terms(term_list, linked_condense=True, unlinked_condense=False)
        expected_result = [[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)], [fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1)]]
        assert function_output == expected_result

        # unlink condensed
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),)
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0))
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0))
            ]
        ]
        function_output = fcc.prepare_condensed_terms(term_list, linked_condense=False, unlinked_condense=True)
        expected_result = fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2)
        assert function_output == expected_result

        # possible alternative pathways

        # omega_list = [
        #     {'rank':3, 'm':1, 'n':2, 'm_h':0, 'n_h':0, 'm_t':[1], 'n_t':[2] },
        #     {'rank':3, 'm':1, 'n':2, 'm_h':0, 'n_h':0, 'm_t':[1], 'n_t':[2] },
        #     {'rank':3, 'm':1, 'n':2, 'm_h':0, 'n_h':0, 'm_t':[1], 'n_t':[2] },
        #     {'rank':3, 'm':1, 'n':2, 'm_h':0, 'n_h':0, 'm_t':[1], 'n_t':[2] },
        #     {'rank':3, 'm':1, 'n':2, 'm_h':0, 'n_h':0, 'm_t':[1], 'n_t':[2] },
        #     {'rank':3, 'm':1, 'n':2, 'm_h':0, 'n_h':0, 'm_t':[1], 'n_t':[2] },
        #     {'rank':3, 'm':1, 'n':2, 'm_h':0, 'n_h':0, 'm_t':[1], 'n_t':[2] },
        # ]

        # tuple_list = [
        #     ({'m_h': 0, 'n_h': 0, 'm_o': 2, 'n_o': 1}, )
        #     (
        #         {'m_h': 0, 'n_h': 0, 'm_o': 2, 'n_o': 1}  # disconnected
        #         {'m_h': 1, 'n_h': 0, 'm_o': 0, 'n_o': 0}  # connected
        #     ),
        # ]
        #  with open(path, 'r') as fp:
        #     test_args = fp.read()
        # slap this whole bad boy into a file and just import term_list
        # omega rank 3
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1], n_t=[2]),
                fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1), ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 2]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=0, n=3, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 3]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=3, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 2]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=2, n_h=1, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[0, 2], n_t=[0, 0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=0, m_t=[0, 2], n_t=[0, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=1, n_h=2, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0], n_t=[2, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=3, n=0, m_o=0, n_o=0, m_t=[0, 3], n_t=[0, 0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=3, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0, 0], n_t=[0, 1, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=0, n=3, m_o=0, n_o=0, m_t=[0, 0, 0], n_t=[0, 1, 2]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[0, 1, 0], n_t=[0, 0, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=0, m_t=[0, 0, 1], n_t=[0, 1, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=0, m_t=[0, 1, 0], n_t=[0, 0, 2]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[0, 1, 1], n_t=[0, 0, 0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=0, m_t=[0, 1, 1], n_t=[0, 0, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=0, m_t=[0, 2, 0], n_t=[0, 0, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                ),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=3, m=1, n=2, m_h=0, n_h=0, m_t=[1, 0, 0], n_t=[2, 0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=3, n=0, m_o=0, n_o=0, m_t=[0, 1, 2], n_t=[0, 0, 0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                    fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),
                ),
            ],
        ]
        function_output = fcc.prepare_condensed_terms(term_list, linked_condense=True, unlinked_condense=False)
        expected_result = [[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1)]]
        assert function_output == expected_result

    def test_simplify_full_cc_python_prefactor(self):
        numerator_list = []
        denominator_list = ['2!', '2!']
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['2!', '2!'])
        assert function_output == expected_result

        numerator_list = ['2!']
        denominator_list = ['2!']
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], [])
        assert function_output == expected_result

        numerator_list = ['2!']
        denominator_list = ['3!']
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['2!'], ['3!'])
        assert function_output == expected_result

        numerator_list = ['2!', '2!', '2!']
        denominator_list = ['2!', '2!']
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['2!'], [])
        assert function_output == expected_result

        numerator_list = ['2!', '2!']
        denominator_list = ['2!', '2!', '2!']
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['2!'])
        assert function_output == expected_result

    def test_build_latex_prefactor(self):
        # single h
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])
        t_list = (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''
        assert function_output == expected_result

        # t_list len>1
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0))
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = r'\frac{1}{2!2!}'
        assert function_output == expected_result

        # x>1 case
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0))
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = r'\frac{1}{2!}'
        assert function_output == expected_result

        # h.m>1 case and h.n > 1 case
        h = fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[2], n_t=[0])
        t_list = (fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),)
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = r'\frac{1}{2!}'
        assert function_output == expected_result

        # case where numerator == '1' and denominator == '1'
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''
        assert function_output == expected_result

    def test_linked_condensed_adjust_t_terms(self):
        """ x """
        # basic
        common_linked_factor_list = [
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)],
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
            ]
        ]
        h = fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0])
        t_list = (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
        function_output = fcc._linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)
        expected_result = (1, [], {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 2})
        assert function_output == expected_result

        # case where 0 == h.m_o == h.n_o
        common_linked_factor_list = [
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)],
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
            ]
        ]
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 0])
        t_list = (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0), fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0))
        function_output = fcc._linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)
        expected_result = (0, [], {'summation_lower': 0, 'summation_upper': 0, 'unlinked': 2})
        assert function_output == expected_result

    def test_creates_f_prefactor(self):
        omega = fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0])
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])
        function_output = fcc._creates_f_prefactor(omega, h)
        expected_result = False
        assert function_output == expected_result

    def test_creates_fbar_prefactor(self):
        omega = fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0])
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])
        function_output = fcc._creates_fbar_prefactor(omega, h)
        assert function_output is False

    def test_make_latex(self):
        # basic
        rank = 0
        term_list = []
        function_output = fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=False)
        expected_result = "()"
        assert function_output == expected_result

        # linked_condense
        rank = 1
        term_list = [[fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]), fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)]]
        function_output = fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=True, print_prefactors=True, color=True)
        expected_result = r'(\disconnected{\bh_0})\bt^{\blue{}\red{z}}_{}'
        assert function_output == expected_result

        # unlink_condense
        rank = 2
        term_list = [[fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=0),)]]
        function_output = fcc._make_latex(rank, term_list, linked_condense=True, unlinked_condense=False, print_prefactors=True, color=True)
        expected_result = r'(\bar{f}\bh^{\blue{}\red{x}}_{\blue{}\red{}})\bt^{\blue{}\red{zy}}_{}'
        assert function_output == expected_result

        # _creates_f_prefactor(omega, h)==True
        rank = 1
        term_list = [[fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=1, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=1, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=0, n_h=0, m_t=[1], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),)], [fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=0, n_h=0, m_t=[1], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]), (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),)]]
        function_output = fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=True)
        expected_result = r'(f\bh^{\blue{}\red{}}_{\blue{}\red{z}} + \bh^{\blue{}\red{}}_{\blue{i}\red{}}\bt^{\blue{i}\red{}}_{\blue{}\red{z}} + \bh^{\blue{i}\red{}}_{\blue{}\red{}}\bt^{}_{\blue{i}\red{z}})'
        assert function_output == expected_result

        # _creates_f_prefactor(omega, h)==False
        rank = 2
        term_list = [[fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=2, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=2, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)]]
        function_output = fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=True)
        expected_result = r'(f^{2}\bh^{\blue{}\red{}}_{\blue{}\red{zy}})'
        assert function_output == expected_result

        # _creates_fbar_prefactor(omega, h)==False
        rank = 2
        term_list = [[fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=2, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=2, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]), fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=0, m_t=[0], n_t=[1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=0, m_t=[0, 0], n_t=[1, 1]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0))]]
        function_output = fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=True)
        expected_result = r'(\bar{f}^{2}\bh^{\blue{}\red{zy}}_{\blue{}\red{}} + \bar{f}\bh^{\blue{}\red{z}}_{\blue{i}\red{}}\bt^{\blue{i}\red{y}}_{} + \frac{1}{2!2!}\bh^{\blue{}\red{}}_{\blue{ij}\red{}}\bt^{\blue{i}\red{z}}_{}\bt^{\blue{j}\red{y}}_{})'
        assert function_output == expected_result

        # long running line else catch
        rank = 2
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=1, n_h=1, m_t=[0], n_t=[0]
                ),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=1, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0], n_t=[1]
                ),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=1, m_t=[0], n_t=[1]),
                (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1], n_t=[0]
                ),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=0, m_t=[0], n_t=[1]),
                (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0], n_t=[1]
                ),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=1, m_t=[1], n_t=[0]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0),),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=1, n_h=1, m_t=[0], n_t=[0]
                ),
                fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=1, n_o=1, m_t=[0], n_t=[1]),
                (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1], n_t=[0]
                ),
                fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=1, n_o=0, m_t=[1], n_t=[0]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=1, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=1, n_o=1, m_t=[1], n_t=[0]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[1, 0], n_t=[0, 1]
                ),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
                (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[0, 1]
                ),
                fcc.connected_h_operator_namedtuple(rank=3, m=0, n=3, m_o=0, n_o=1, m_t=[0, 0], n_t=[1, 1]),
                (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(
                    rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[1, 0], n_t=[0, 1]
                ),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[1, 0], n_t=[0, 1]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[0, 1], n_t=[1, 0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[1, 0], n_t=[0, 1]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[1, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=1, m_t=[1, 0], n_t=[0, 1]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[0, 1], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=1, n_o=0, m_t=[0, 0], n_t=[1, 1]),
                (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[0, 1]),
                fcc.connected_h_operator_namedtuple(rank=3, m=1, n=2, m_o=0, n_o=1, m_t=[1, 0], n_t=[0, 1]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=0, m_t=[1, 0], n_t=[0, 1]),
                fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[1, 1], n_t=[0, 0]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[1, 0], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=1, n_o=0, m_t=[1, 0], n_t=[0, 1]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[0, 1], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=1, n_o=0, m_t=[1, 0], n_t=[0, 1]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=1, n_h=0, m_t=[0, 0], n_t=[0, 1]),
                fcc.connected_h_operator_namedtuple(rank=3, m=2, n=1, m_o=0, n_o=1, m_t=[1, 1], n_t=[0, 0]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=1, n_o=0)),
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=1, n=1, m_h=0, n_h=1, m_t=[0, 1], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=3, m=3, n=0, m_o=1, n_o=0, m_t=[1, 1], n_t=[0, 0]),
                (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1)),
            ],
        ]
        function_output = fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=False)
        # TODO very picky r string can be fixed later by adding r strings and strings with /n
        expected_result = r'''(
f\bar{f}\bh^{\blue{}\red{z}}_{\blue{}\red{y}} + f\frac{1}{2!}\bh^{\blue{}\red{}}_{\blue{i}\red{z}}\bt^{\blue{i}\red{y}}_{} + \bar{f}\bh^{\blue{}\red{z}}_{\blue{i}\red{}}\bt^{\blue{i}\red{}}_{\blue{}\red{y}} + f\bh^{\blue{i}\red{}}_{\blue{}\red{z}}\bt^{\blue{}\red{y}}_{\blue{i}\red{}} + f\bar{f}\frac{1}{2!}\bh^{\blue{}\red{z}}_{\blue{i}\red{y}}\bt^{\blue{i}\red{}}_{} + \bar{f}\frac{1}{2!}\bh^{\blue{i}\red{z}}_{\blue{}\red{}}\bt^{}_{\blue{i}\red{y}} + f\bar{f}\frac{1}{2!}\bh^{\blue{i}\red{z}}_{\blue{}\red{y}}\bt^{}_{\blue{i}\red{}}
    \\  &+  % split long equation
\frac{1}{2!}\bh^{\blue{}\red{}}_{\blue{ij}\red{}}\bt^{\blue{i}\red{}}_{\blue{}\red{z}}\bt^{\blue{j}\red{y}}_{} + f\frac{1}{3!}\bh^{\blue{}\red{}}_{\blue{ij}\red{z}}\bt^{\blue{i}\red{}}_{}\bt^{\blue{j}\red{y}}_{} + \bh^{\blue{i}\red{}}_{\blue{j}\red{}}\bt^{}_{\blue{i}\red{z}}\bt^{\blue{j}\red{y}}_{} + \bh^{\blue{i}\red{}}_{\blue{j}\red{}}\bt^{\blue{}\red{z}}_{\blue{i}\red{}}\bt^{\blue{j}\red{}}_{\blue{}\red{y}} + f\frac{1}{2!}\bh^{\blue{i}\red{}}_{\blue{j}\red{z}}\bt^{\blue{}\red{y}}_{\blue{i}\red{}}\bt^{\blue{j}\red{}}_{} + \bar{f}\frac{1}{2!}\bh^{\blue{}\red{z}}_{\blue{ij}\red{}}\bt^{\blue{i}\red{}}_{}\bt^{\blue{j}\red{}}_{\blue{}\red{y}} + f\frac{1}{2!}\bh^{\blue{i}\red{}}_{\blue{j}\red{z}}\bt^{}_{\blue{i}\red{}}\bt^{\blue{j}\red{y}}_{}
)'''
        assert function_output == expected_result

    def test_write_cc_latex_from_lists(self):
        # basic test
        rank = 1
        fully = [[fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=1, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)]]
        linked = []
        unlinked = [[fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]), fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)]]
        function_output = fcc._write_cc_latex_from_lists(rank, fully, linked, unlinked)
        expected_result = r'(\bar{f}\bh^{\blue{}\red{z}}_{\blue{}\red{}})' + '\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n%\n%\n\\\\  &+\n%\n%\n' + r'(\bh_0)\bt^{\blue{}\red{z}}'
        assert function_output == expected_result

    def test_write_cc_latex_from_lists_rank_zero_case(self):
        # rank 0 test case
        rank = 0
        fully = [[fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]), (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),)]]
        linked = []
        unlinked = []
        function_output = fcc._write_cc_latex_from_lists(rank, fully, linked, unlinked)
        expected_result = r'(\bh_0 + \bh^{\blue{}\red{}}_{\blue{i}\red{}}\bt^{\blue{i}\red{}} + \bh^{\blue{i}\red{}}_{\blue{}\red{}}\bt_{\blue{i}\red{}}) + () + ()'
        assert function_output == expected_result

    def test_write_cc_latex_from_lists_high_rank_case(self):
        rank = 2
        fully = [[fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=2, n_h=0, m_t=[0], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=2, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=1, m_t=[0], n_t=[1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=1, m_t=[1], n_t=[0]), (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]), (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[1, 0], n_t=[0, 1]), (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[1, 1], n_t=[0, 0]), (fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1))]]
        linked = [[fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=1, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1), fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1, 0], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=1, m_t=[0, 0], n_t=[0, 1]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[1, 1], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=1, n_h=0, m_t=[1, 0], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=1, m_t=[0, 1], n_t=[0, 0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0))]]
        unlinked = [[fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2], n_t=[0]), fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),)], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 2]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=2, n_h=0, m_o=0, n_o=0))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 1]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=1, n_h=1, m_o=0, n_o=0))], [fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]), fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[0, 2], n_t=[0, 0]), (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2), fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0))]]
        function_output = fcc._write_cc_latex_from_lists(rank, fully, linked, unlinked)
        expected_result1 = r'(f^{2}\bh^{\blue{}\red{}}_{\blue{}\red{zy}} + f\frac{1}{2!}\bh^{\blue{}\red{}}_{\blue{i}\red{z}}\bt^{\blue{i}\red{}}_{\blue{}\red{y}} + f\bh^{\blue{i}\red{}}_{\blue{}\red{z}}\bt_{\blue{i}\red{y}} + \frac{1}{2!2!}\bh^{\blue{}\red{}}_{\blue{ij}\red{}}\bt^{\blue{i}\red{}}_{\blue{}\red{z}}\bt^{\blue{j}\red{}}_{\blue{}\red{y}} + \bh^{\blue{i}\red{}}_{\blue{j}\red{}}\bt_{\blue{i}\red{z}}\bt^{\blue{j}\red{}}_{\blue{}\red{y}} + \frac{1}{2!2!}\bh^{\blue{ij}\red{}}_{\blue{}\red{}}\bt_{\blue{i}\red{z}}\bt_{\blue{j}\red{y}})'
        expected_result2 = r'(\frac{1}{2!}\bh_0)\bt_{\blue{}\red{z}}\bt_{\blue{}\red{y}}'
        expected_result3 = r'(f\bh^{\blue{}\red{}}_{\blue{}\red{y}} + \frac{1}{2!}\bh^{\blue{}\red{}}_{\blue{i}\red{}}\bt^{\blue{i}\red{}}_{\blue{}\red{y}} + f\frac{1}{2!2!}\bh^{\blue{}\red{}}_{\blue{i}\red{y}}\bt^{\blue{i}\red{}} + \frac{1}{2!}\bh^{\blue{i}\red{}}_{\blue{}\red{}}\bt_{\blue{i}\red{y}} + f\frac{1}{2!}\bh^{\blue{i}\red{}}_{\blue{}\red{y}}\bt_{\blue{i}\red{}})\bt_{\blue{}\red{z}}'
        expected_result4 = r'(\bh_0 + \frac{1}{2!}\bh^{\blue{}\red{}}_{\blue{i}\red{}}\bt^{\blue{i}\red{}} + \frac{1}{2!2!}\bh^{\blue{}\red{}}_{\blue{ij}\red{}}\bt^{\blue{ij}\red{}} + \frac{1}{2!}\bh^{\blue{i}\red{}}_{\blue{}\red{}}\bt_{\blue{i}\red{}} + \frac{1}{2!}\bh^{\blue{i}\red{}}_{\blue{j}\red{}}\bt^{\blue{j}\red{}}_{\blue{i}\red{}} + \frac{1}{2!2!}\bh^{\blue{ij}\red{}}_{\blue{}\red{}}\bt_{\blue{ij}\red{}})\bt_{\blue{}\red{zy}}'
        expected_result = expected_result1 + '\n%\n%\n\\\\  &+\n%\n%\n' + expected_result2 + '\n    \\\\  &+  % split long equation\n' + expected_result3 + '\n%\n%\n\\\\  &+\n%\n%\n' + expected_result4
        assert function_output == expected_result

    def test_generate_cc_latex_equations(self):
        omega = fcc.general_operator_namedtuple(name='d', rank=1, m=1, n=0)
        H = fcc.hamiltonian_namedtuple(maximum_rank=1, operator_list=[fcc.h_operator_namedtuple(rank=0, m=0, n=0), fcc.h_operator_namedtuple(rank=1, m=0, n=1), fcc.h_operator_namedtuple(rank=1, m=1, n=0)])
        s_taylor_expansion = [fcc.general_operator_namedtuple(name='1', rank=0, m=0, n=0), [fcc.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1), fcc.general_operator_namedtuple(name='s^1', rank=1, m=1, n=0)]]
        function_output = fcc._generate_cc_latex_equations(omega, H, s_taylor_expansion, remove_f_terms=True)
        expected_result = r'    \textit{no fully connected terms}' + '\n%\n%\n\\\\  &+\n%\n%\n' + r'    \textit{no linked disconnected terms}' + '\n%\n%\n\\\\  &+\n%\n%\n' + r'(\bh_0)\bt_{\blue{}\red{z}}'
        assert function_output == expected_result

    def test_generate_left_hand_side(self):
        omega = fcc.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
        function_output = fcc._generate_left_hand_side(omega)
        expected_result = r'i\left(\dv{\bt^{}_{ij}}{\tau} + \dv{\bt^{}_{i}}{\tau}\bt^{}_{j} + \bt^{}_{i}\dv{\bt^{}_{j}}{\tau} + \bt^{}_{ij}\varepsilon + \bt^{}_{i}\bt^{}_{j}\varepsilon\right)'
        assert function_output == expected_result

    def test_generate_left_hand_side_omega_zero_case(self):
        omega = fcc.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        function_output = fcc._generate_left_hand_side(omega)
        expected_result = r'i\left(\varepsilon\right)'
        assert function_output == expected_result

    def test_wrap_align_environment_b_omega(self):
        omega = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        rank_name = 'LINEAR'
        lhs = 'i\\left(\\dv{\\bt^{i}_{}}{\\tau} + \\bt^{i}_{}\\varepsilon\\right)'
        eqns = '(\\bar{f}\\bh^{\\blue{}\\red{z}}_{\\blue{}\\red{}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt^{\\blue{}\\red{z}}'
        function_output = fcc._wrap_align_environment(omega, rank_name, lhs, eqns)
        expected_result = '\\begin{align}\\begin{split}\n    \\hat{\\Omega} = \\down{i}\n\\\\ LHS &=\n    i\\left(\\dv{\\bt^{i}_{}}{\\tau} + \\bt^{i}_{}\\varepsilon\\right)\n\\\\ RHS &=\n%\n%\n(\\bar{f}\\bh^{\\blue{}\\red{z}}_{\\blue{}\\red{}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt^{\\blue{}\\red{z}}\n\\end{split}\\end{align}\n\n'
        assert function_output == expected_result
    
    def test_wrap_align_environment_d_omega(self):
        omega = fcc.general_operator_namedtuple(name='d', rank=1, m=1, n=0)
        rank_name = 'LINEAR'
        lhs = 'i\\left(\\dv{\\bt^{}_{i}}{\\tau} + \\bt^{}_{i}\\varepsilon\\right)'
        eqns = '(f\\bh^{\\blue{}\\red{}}_{\\blue{}\\red{z}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt_{\\blue{}\\red{z}}'
        function_output = fcc._wrap_align_environment(omega, rank_name, lhs, eqns)
        expected_result = '\\begin{align}\\begin{split}\n    \\hat{\\Omega} = \\up{i}\n\\\\ LHS &=\n    i\\left(\\dv{\\bt^{}_{i}}{\\tau} + \\bt^{}_{i}\\varepsilon\\right)\n\\\\ RHS &=\n%\n%\n(f\\bh^{\\blue{}\\red{}}_{\\blue{}\\red{z}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt_{\\blue{}\\red{z}}\n\\end{split}\\end{align}\n\n'
        assert function_output == expected_result

    def test_wrap_align_environment_blank_omega(self):
        omega = fcc.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        rank_name = '0 order'
        lhs = 'i\\left(\\varepsilon\\right)'
        eqns = '(\\bh_0 + \\bh^{\\blue{}\\red{}}_{\\blue{i}\\red{}}\\bt^{\\blue{i}\\red{}} + \\bh^{\\blue{i}\\red{}}_{\\blue{}\\red{}}\\bt_{\\blue{i}\\red{}}) + () + ()'
        function_output = fcc._wrap_align_environment(omega, rank_name, lhs, eqns)
        expected_result = '\\begin{align}\\begin{split}\n    \\hat{\\Omega} = 1\n\\\\ LHS &=\n    i\\left(\\varepsilon\\right)\n\\\\ RHS &=\n%\n%\n(\\bh_0 + \\bh^{\\blue{}\\red{}}_{\\blue{i}\\red{}}\\bt^{\\blue{i}\\red{}} + \\bh^{\\blue{i}\\red{}}_{\\blue{}\\red{}}\\bt_{\\blue{i}\\red{}}) + () + ()\n\\end{split}\\end{align}\n\n'
        assert function_output == expected_result

    def test_generate_full_cc_latex(self):
        # runs function for coverage
        fcc.generate_full_cc_latex([2, 2, 2, 2], only_ground_state=False, path="./generated_latex.txt")
        fcc.generate_full_cc_latex([2, 2, 2, 2], only_ground_state=True, path="./generated_latex.txt")
