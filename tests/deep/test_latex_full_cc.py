# system imports
# import re
import pytest
from os.path import abspath, dirname, join
deep_dir = dirname(abspath(__file__))
root_dir = join(deep_dir, 'files')
classtest = 'test_latex_full_cc'
# local imports
from . import context
from . import large_test_data
import latex_full_cc as fcc
import namedtuple_defines as nt

# TODO add functions where multiple tests are encased in one function for future debugging
# TODO fix long lists


class Test_generate_omega_operator:

    def test_basic(self):
        """basic test"""

        # run function
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


class Test_generate_full_cc_hamiltonian_operator:

    def test_basic(self):
        """basic test"""

        # run function
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


class Test_generate_s:

    def test_basic(self):
        """basic test"""

        # run function
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


class Test_t_joining_with_t_terms:

    def test_basic(self, omega_zero, h_0, s_term_identity):
        """ x """

        # input data
        args = [
            omega_zero,
            h_0,
            [(s_term_identity), ],
        ]

        assert fcc._t_joining_with_t_terms(*args, nof_creation_ops=1) is False

        # change s term
        args[2] = [nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1), ]

        assert fcc._t_joining_with_t_terms(*args, nof_creation_ops=1) is True


class Test_omega_joining_with_itself:

    def test_omega_zero(self, omega_db, omega_zero, h_0, s_term_identity):
        """ (omega.m == 0) or (omega.n == 0) return False """

        # input data
        args = (
            omega_zero,
            h_0,
            [s_term_identity, ],
        )

        assert fcc._omega_joining_with_itself(*args) is False

    def test_omega_nonzero(self, omega_db, omega_zero, h_0, s_term_identity):
        """ (omega.n > 0 and h.m > 0) or (omega.m > 0 and h.n > 0) return False """

        # input data
        args = (
            omega_db,
            fcc.h_operator_namedtuple(rank=2, m=1, n=1),
            [s_term_identity, ],
        )

        assert fcc._omega_joining_with_itself(*args) is False

    def test_omega_nonzero_false_case(self, omega_db, omega_zero, h_0, s_term_identity):
        """ for s in s_list: if (omega.n > 0 and s.m > 0) or (omega.m > 0 and s.n > 0): return False """

        # input data
        args = (
            omega_db,
            h_0,
            [nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1), ],
        )

        assert fcc._omega_joining_with_itself(*args) is False

    def test_omega_nonzero_true_case(self, omega_db, omega_zero, h_0, s_term_identity):
        """ else == True """

        # input data
        args = (
            omega_db,
            h_0,
            [s_term_identity, ],
        )

        assert fcc._omega_joining_with_itself(*args) is True


class Test_h_joining_with_itself:

    def test_zero_case(self, omega_zero, h_0, s_term_identity):
        """(h.m == 0) or (h.n == 0): return False"""

        # input data
        omega = omega_zero
        h = h_0
        s_list = [s_term_identity, ]

        # run function
        function_output = fcc._h_joining_with_itself(omega, h, s_list)

        assert function_output is False

    def test_nonzero_case(self, omega_zero, h_0, s_term_identity):
        """if (omega.n > 0 and h.m > 0) or (omega.m > 0 and h.n > 0): return False"""

        # input data
        omega = nt.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list = [nt.general_operator_namedtuple(name='s^1', rank=1, m=1, n=0)]

        # run function
        function_output = fcc._h_joining_with_itself(omega, h, s_list)

        assert function_output is False

    def test_false_case(self, omega_zero, h_0, s_term_identity):
        """for s in s_list: if (omega.n > 0 and s.m > 0) or (omega.m > 0 and s.n > 0): return False"""

        # input data
        omega = omega_zero
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list = [nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1)]

        # run function
        function_output = fcc._h_joining_with_itself(omega, h, s_list)

        assert function_output is False

    def test_true_case(self, omega_zero, h_0, s_term_identity):
        """else == True"""

        # input data
        omega = omega_zero
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_list = [s_term_identity, ]

        # run function
        function_output = fcc._h_joining_with_itself(omega, h, s_list)

        assert function_output is True


class Test_generate_valid_s_n_operator_permutations:

    def test_basic(self, omega_zero, h_0, s_term_identity):
        """basic test"""

        # input data
        omega = omega_zero
        h = h_0
        s_series_term = [[s_term_identity, ], ]

        # run function
        function_output = fcc._generate_valid_s_n_operator_permutations(omega, h, s_series_term)
        expected_result = [[s_term_identity, ], ]

        assert expected_result == function_output


class Test_generate_all_valid_t_connection_permutations:

    def test_basic(self):
        """basic test"""

        # input data
        omega_2_0 = nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
        h_1_1 = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        s_term_list = [
            nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
            nt.general_operator_namedtuple(name='s^1_1', rank=2, m=1, n=1),
        ]

        # run function
        function_output = fcc._generate_all_valid_t_connection_permutations(
            omega_2_0, h_1_1, s_term_list, log_invalid=True
        )
        expected_result = ([((0, 0), (0, 1))], [((1, 1), (1, 0)), ((2, 0), (0, 1))])

        assert expected_result == function_output


class Test_generate_all_omega_h_connection_permutations:

    def test_basic(self, omega_zero, h_0, s_term_identity):
        """basic test"""

        # input data
        omega = nt.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)
        h = fcc.h_operator_namedtuple(rank=2, m=2, n=0)
        valid_permutations = [
            [
                nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                nt.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ]
        ]

        # run function
        function_output = fcc._generate_all_omega_h_connection_permutations(
            omega,
            h,
            valid_permutations,
            found_it_bool=False
        )
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


class Test_remove_duplicate_s_permutations:

    def test_basic(self):  # TODO add dupe test
        """basic test"""

        # input data
        s_list = [
            [
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            ],
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
            ]
        ]

        # run function
        function_output = fcc._remove_duplicate_s_permutations(s_list)
        expected_result = {
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
            ),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
            )
        }

        assert expected_result == function_output


class Test_generate_explicit_connections:

    def test_basic(self, omega_zero):

        # input data
        omega = omega_zero
        h = fcc.h_operator_namedtuple(rank=2, m=0, n=2)
        unique_s_permutations = {
            (
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            )
        }

        # run function
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
    #     h = fix

        # TODO if req, also add raise excepts for if states?
        # invalid term case 1
        # invalid term case 2


class Test_remove_f_zero_terms:

    def test_basic(self):
        # TODO exclude log replace with raise exceptions
        """basic test"""

        # input data
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

        # run function
        function_output = fcc._remove_f_zero_terms(labeled_permutations)

        assert function_output == labeled_permutations


class Test_filter_out_valid_s_terms:

    def test_basic(self, omega_zero):
        # TODO test list mutation
        """basic test"""

        # input data
        omega = omega_zero
        H = fcc.generate_full_cc_hamiltonian_operator(maximum_rank=2)

        s_series_term = nt.general_operator_namedtuple(name='1', rank=0, m=0, n=0)
        term_list = []
        total_list = []

        # run function
        function_output = fcc._filter_out_valid_s_terms(
            omega,
            H,
            s_series_term,
            term_list,
            total_list,
            remove_f_terms=True
        )

        assert function_output is None


class Test_seperate_s_terms_by_connection:

    def test_basic(self):
        # TODO add further tests
        """basic test"""

        # input data
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

        # run function
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


class Test_build_h_term_latex_labels_normal:

    def test_basic(self):
        """basic test"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])

        # run function
        function_output = fcc._build_h_term_latex_labels(h, condense_offset=0, color=True)
        expected_result = '\\bh^' + r'{\blue{}\red{}}_'r'{\blue{i}\red{}}'

        assert function_output == expected_result

    def test_rank_zero_case(self):
        """h.rank==0"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])

        # run function
        function_output = fcc._build_h_term_latex_labels(h, condense_offset=0, color=True)
        expected_result = '\\bh_0'

        assert function_output == expected_result


class Test_build_t_term_latex_labels:

    def test_zero_case(self):
        """term all zero"""

        # input data
        term = fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}

        # run function
        function_output = fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        expected_result = '^{}_{}'

        assert function_output == expected_result

    def test_case_1(self):
        """x"""

        # input data
        term = fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 1, 'summation_lower': 0, 'unlinked': 0}

        # run function
        function_output = fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        expected_result = '^{}_' + r'{\blue{i}\red{}}'

        assert function_output == expected_result

    def test_case_2(self):
        """x"""

        # input data
        term = fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
        offset_dict = {'summation_upper': 0, 'summation_lower': 0, 'unlinked': 0}

        # run function
        function_output = fcc._build_t_term_latex_labels(term, offset_dict, color=True)
        expected_result = r'^{\blue{i}\red{}}_{}'

        assert function_output == expected_result


class Test_build_t_term_latex:

    def test_basic(self):
        """basic test"""

        # input data
        s = fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)

        # run function
        function_output = fcc._build_t_term_latex(s, h=None)
        expected_result = r'\bt^{\blue{}\red{z}}_{}'

        assert function_output == expected_result

    def test_non_none_h(self):
        """h!=none"""

        # input data
        s = fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])

        # run function
        function_output = fcc._build_t_term_latex(s, h)
        expected_result = r'\bt^{\blue{}\red{z}}_{}'

        assert function_output == expected_result


class Test_validate_s_terms:

    def test_valid_term(self):
        """term is valid"""

        # input data
        s_list = (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)

        # run function
        function_output = fcc._validate_s_terms(s_list)
        expected_result = None

        assert function_output == expected_result

    def test_invalid_term(self):
        """term is invalid"""

        # input data
        s_list = ("weeee")

        # exception check
        with pytest.raises(AssertionError):
            fcc._validate_s_terms(s_list)


class Test_generate_linked_common_terms:

    def test_basic(self):
        """test basic"""

        # input data
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
            ]
        ]

        # run function
        function_output = fcc._generate_linked_common_terms(term_list)
        expected_result = [[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)]]

        assert function_output == expected_result


class Test_prepare_condensed_terms:

    def test_link_condensed(self):
        """linked condense == True"""

        # input data
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

        # run function
        function_output = fcc.prepare_condensed_terms(term_list, linked_condense=True, unlinked_condense=False)
        expected_result = [
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
            ],
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=1)
            ]
        ]

        assert function_output == expected_result

    def test_unlink_condensed(self):
        """unlinked condense == True"""

        # input data
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=0, n_h=0, m_t=[2, 0], n_t=[0, 0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[0, 1], n_t=[0, 0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=2),
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0)
                )
            ]
        ]

        # run function
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

    def test_large_term_list(self):
        """large term list to trigger the long lines if statement"""

        # input data
        term_list = large_test_data.prepare_condensed_terms.large_term_list

        # run function
        function_output = fcc.prepare_condensed_terms(term_list, linked_condense=True, unlinked_condense=False)
        expected_result = [[fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=1)]]

        assert function_output == expected_result


class Test_simplify_full_cc_python_prefactor:

    def test_num_empty(self):
        """numerator list empty"""

        # input data
        numerator_list = []
        denominator_list = ['2!', '2!']

        # run function
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['2!', '2!'])

        assert function_output == expected_result

    def test_num_eq_denom(self):
        """numerator == denominator"""

        # input data
        numerator_list = ['2!']
        denominator_list = ['2!']

        # run function
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], [])

        assert function_output == expected_result

    def test_num_not_eq_denom(self):
        """numerator != denominator"""

        # input data
        numerator_list = ['2!']
        denominator_list = ['3!']

        # run function
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['2!'], ['3!'])

        assert function_output == expected_result

    def test_more_num_than_denom(self):
        """len(numerator_list) > len(denominator_list)"""

        # input data
        numerator_list = ['2!', '2!', '2!']
        denominator_list = ['2!', '2!']

        # run function
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = (['2!'], [])

        assert function_output == expected_result

    def test_more_denom_than_num(self):
        """len(numerator_list) < len(denominator_list)"""

        # input data
        numerator_list = ['2!', '2!']
        denominator_list = ['2!', '2!', '2!']

        # run function
        function_output = fcc._simplify_full_cc_python_prefactor(numerator_list, denominator_list)
        expected_result = ([], ['2!'])

        assert function_output == expected_result


class Test_build_latex_prefactor:

    def test_single_h(self):
        """x"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])
        t_list = (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),)

        # run function
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''

        assert function_output == expected_result

    def test_t_list_long(self):
        """len(t_list) > 1"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
        )

        # run function
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = r'\frac{1}{2!2!}'

        assert function_output == expected_result

    def test_x_greater_than_1(self):
        """x > 1"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1])
        t_list = (
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
            fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
        )

        # run function
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = r'\frac{1}{2!}'

        assert function_output == expected_result

    def test_h_non_zero_nm(self):
        """h.m>1 case and h.n > 1 case"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=0, n_o=0, m_t=[2], n_t=[0])
        t_list = (fcc.connected_namedtuple(m_h=0, n_h=2, m_o=0, n_o=0),)

        # run function
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = r'\frac{1}{2!}'

        assert function_output == expected_result

    def test_num_and_denom_eq_1(self):
        """case where numerator == '1' and denominator == '1'"""

        # input data
        h = fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1])
        t_list = (fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),)

        # run function
        function_output = fcc._build_latex_prefactor(h, t_list, simplify_flag=True)
        expected_result = ''

        assert function_output == expected_result


class Test_linked_condensed_adjust_t_terms:

    def test_basic(self):
        """ x """

        # input data
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

        # run function
        function_output = fcc._linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)
        expected_result = (
            1,
            [],
            {
                'summation_upper': 0,
                'summation_lower': 0,
                'unlinked': 2
            }
        )

        assert function_output == expected_result

    def test_h_zero_if(self):
        """case where 0 == h.m_o == h.n_o"""

        # input data
        common_linked_factor_list = [
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)],
            [
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
            ]
        ]
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 0])
        t_list = (
            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
            fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
        )

        # run function
        function_output = fcc._linked_condensed_adjust_t_terms(common_linked_factor_list, h, t_list)
        expected_result = (
            0,
            [],
            {
                'summation_lower': 0,
                'summation_upper': 0,
                'unlinked': 2
            }
        )

        assert function_output == expected_result


class Test_creates_f_prefactor:

    def test_basic(self):
        """basic test"""

        # input data
        omega = fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0])
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])

        # run function
        function_output = fcc._creates_f_prefactor(omega, h)
        expected_result = False

        assert function_output == expected_result


class Test_creates_fbar_prefactor:

    def test_basic(self):
        """basic test"""

        # input data
        omega = fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0])
        h = fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0])

        # run function
        function_output = fcc._creates_fbar_prefactor(omega, h)

        assert function_output is False


class Test_make_latex:

    def test_basic(self):
        """basic test"""

        # input data
        rank = 0
        term_list = []

        # run function
        function_output = fcc._make_latex(
            rank,
            term_list,
            linked_condense=False,
            unlinked_condense=False,
            print_prefactors=True,
            color=False
        )
        expected_result = "()"

        assert function_output == expected_result

    def test_linked_condense(self):
        """linked_condense == True"""

        # input data
        rank = 1
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]),
                fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                )
            ]
        ]

        # run function
        function_output = fcc._make_latex(
            rank,
            term_list,
            linked_condense=False,
            unlinked_condense=True,
            print_prefactors=True,
            color=True
        )
        expected_result = r'(\disconnected{\bh_0})\bt^{\blue{}\red{z}}_{}'

        assert function_output == expected_result

    def test_unlink_condense(self):
        """unlink_condense == True"""

        # input data
        rank = 2
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=2, n_o=0),
                )
            ]
        ]

        # run function
        function_output = fcc._make_latex(
            rank,
            term_list,
            linked_condense=True,
            unlinked_condense=False,
            print_prefactors=True,
            color=True
        )
        expected_result = r'(\bar{f}\bh^{\blue{}\red{x}}_{\blue{}\red{}})\bt^{\blue{}\red{zy}}_{}'
        assert function_output == expected_result

    def test_true_f_prefactor(self):
        """_creates_f_prefactor(omega, h)==True"""

        # input data
        rank = 1
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=1, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=1, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=0, n_h=0, m_t=[1], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=1),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=1, n=0, m_h=0, n_h=0, m_t=[1], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]),
                (
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=1),
                )
            ]
        ]

        # run function
        function_output = fcc._make_latex(
            rank,
            term_list,
            linked_condense=False,
            unlinked_condense=False,
            print_prefactors=True,
            color=True
        )
        expected_result = str(
            '(f\\bh^{\\blue{}\\red{}}_{\\blue{}\\red{z}} + \\bh^{\\blue{}\\red{}}_{\\blue{i}\\red{}}\\bt^{\\blue{i}\\'
            'red{}}_{\\blue{}\\red{z}} + \\bh^{\\blue{i}\\red{}}_{\\blue{}\\red{}}\\bt^{}_{\\blue{i}\\red{z}})'
        )

        assert function_output == expected_result

    def test_false_f_prefactor(self):
        """_creates_f_prefactor(omega, h)==False"""

        # input data
        rank = 2
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=2, n=0, m_h=2, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=2, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ]
        ]

        # run function
        function_output = fcc._make_latex(rank, term_list, linked_condense=False, unlinked_condense=False, print_prefactors=True, color=True)
        expected_result = '(f^{2}\\bh^{\\blue{}\\red{}}_{\\blue{}\\red{zy}})'

        assert function_output == expected_result

    def test_false_fbar_prefactor(self):
        """_creates_fbar_prefactor(omega, h)==False"""

        # input data
        rank = 2
        term_list = [
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=2, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=2, m=2, n=0, m_o=2, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]),
                fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=0, m_t=[0], n_t=[1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=0, m_t=[0, 0], n_t=[1, 1]),
                fcc.connected_h_operator_namedtuple(rank=2, m=0, n=2, m_o=0, n_o=0, m_t=[0, 0], n_t=[1, 1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0),
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
                )
            ]
        ]

        # run function
        function_output = fcc._make_latex(
            rank,
            term_list,
            linked_condense=False,
            unlinked_condense=False,
            print_prefactors=True,
            color=True
        )
        expected_result = str(
            '(\\bar{f}^{2}\\bh^{\\blue{}\\red{zy}}_{\\blue{}\\red{}} + \\bar{f}\\bh^{\\blue{}\\red{z}}_{\\blue{i}\\red{'
            '}}\\bt^{\\blue{i}\\red{y}}_{} + \\frac{1}{2!2!}\\bh^{\\blue{}\\red{}}_{\\blue{ij}\\red{}}\\bt^{\\blue{i}\\'
            'red{z}}_{}\\bt^{\\blue{j}\\red{y}}_{})'
        )

        assert function_output == expected_result

    def test_long_catch(self):
        """long running line else catch"""

        # input data
        rank = 2
        term_list = large_test_data.make_latex.long_catch_term_list

        # run function
        function_output = fcc._make_latex(
            rank,
            term_list,
            linked_condense=False,
            unlinked_condense=False,
            print_prefactors=True,
            color=False
        )

        # open file
        func_name = "test_long_catch_out.txt"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_write_cc_latex_from_lists:

    def test_basic(self):
        """basic test"""

        # input data
        rank = 1
        fully = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=1, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ]
        ]
        linked = []
        unlinked = [
            [
                fcc.connected_omega_operator_namedtuple(rank=1, m=0, n=1, m_h=0, n_h=0, m_t=[0], n_t=[1]),
                fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                )
            ]
        ]

        # run function
        function_output = fcc._write_cc_latex_from_lists(rank, fully, linked, unlinked)
        expected_result = str(
            '(\\bar{f}\\bh^{\\blue{}\\red{z}}_{\\blue{}\\red{}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconne'
            'cted terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt^{\\blue{}\\red{z}}'
        )

        assert function_output == expected_result

    def test_rank_zero_case(self):
        """rank ==0 """

        # input data
        rank = 0
        fully = [
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0], n_t=[0]),
                (
                    fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=0, n_o=0),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0], n_t=[1]),
                (
                    fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0),
                )
            ],
            [
                fcc.connected_omega_operator_namedtuple(rank=0, m=0, n=0, m_h=0, n_h=0, m_t=[0], n_t=[0]),
                fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=0, n_o=0, m_t=[1], n_t=[0]),
                (
                    fcc.connected_namedtuple(m_h=0, n_h=1, m_o=0, n_o=0),
                )
            ]
        ]
        linked = []
        unlinked = []

        # run function
        function_output = fcc._write_cc_latex_from_lists(rank, fully, linked, unlinked)
        expected_result = str(
            '(\\bh_0 + \\bh^{\\blue{}\\red{}}_{\\blue{i}\\red{}}\\bt^{\\blue{i}\\red{}} + \\bh^{\\blue{i}\\red{}}_{\\bl'
            'ue{}\\red{}}\\bt_{\\blue{i}\\red{}}) + () + ()'
        )

        assert function_output == expected_result

    def test_high_rank_case(self):
        """rank > 1 """

        # input data
        rank = 2
        fully = large_test_data.write_cc_latex_from_lists_high_rank_case.fully
        linked = large_test_data.write_cc_latex_from_lists_high_rank_case.linked
        unlinked = large_test_data.write_cc_latex_from_lists_high_rank_case.unlinked

        # run function
        function_output = fcc._write_cc_latex_from_lists(rank, fully, linked, unlinked)

        # open file
        func_name = "write_cc_latex_from_lists_high_rank_case_out.txt"
        file_name = join(root_dir, classtest, func_name)
        with open(file_name, 'r') as fp:
            expected_result = fp.read()

        assert function_output == expected_result


class Test_generate_cc_latex_equations:

    def test_basic(self):
        """basic test"""

        # input data
        omega = fcc.general_operator_namedtuple(name='d', rank=1, m=1, n=0)
        H = fcc.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                fcc.h_operator_namedtuple(rank=0, m=0, n=0),
                fcc.h_operator_namedtuple(rank=1, m=0, n=1),
                fcc.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        s_taylor_expansion = [
            fcc.general_operator_namedtuple(name='1', rank=0, m=0, n=0),
            [
                fcc.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                fcc.general_operator_namedtuple(name='s^1', rank=1, m=1, n=0)
            ]
        ]

        # run function
        function_output = fcc._generate_cc_latex_equations(omega, H, s_taylor_expansion, remove_f_terms=True)
        expected_result = str(
            '    \\textit{no fully connected terms}\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n'
            '%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt_{\\blue{}\\red{z}}'
        )

        assert function_output == expected_result


class Test_generate_left_hand_side:

    def test_basic(self):
        """basic test"""

        # input data
        omega = fcc.general_operator_namedtuple(name='dd', rank=2, m=2, n=0)

        # run function
        function_output = fcc._generate_left_hand_side(omega)
        expected_result = str(
            'i\\left(\\dv{\\bt^{}_{ij}}{\\tau} + \\dv{\\bt^{}_{i}}{\\tau}\\bt^{}_{j} + \\bt^{}_{i}\\dv{\\bt^{}_{j}}{\\t'
            'au} + \\bt^{}_{ij}\\varepsilon + \\bt^{}_{i}\\bt^{}_{j}\\varepsilon\\right)'
        )

        assert function_output == expected_result

    def test_omega_zero_case(self):
        """omega rank,m, and n == 0"""

        # input data
        omega = fcc.general_operator_namedtuple(name='', rank=0, m=0, n=0)

        # run function
        function_output = fcc._generate_left_hand_side(omega)
        expected_result = 'i\\left(\\varepsilon\\right)'

        assert function_output == expected_result


class Test_wrap_align_environment:

    def test_b_omega(self):
        """omega.name=='b' """

        # input data
        omega = fcc.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        rank_name = 'LINEAR'
        lhs = 'i\\left(\\dv{\\bt^{i}_{}}{\\tau} + \\bt^{i}_{}\\varepsilon\\right)'
        eqns = str(
            '(\\bar{f}\\bh^{\\blue{}\\red{z}}_{\\blue{}\\red{}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconne'
            'cted terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt^{\\blue{}\\red{z}}'
        )

        # run function
        function_output = fcc._wrap_align_environment(omega, rank_name, lhs, eqns)
        expected_result = str(
            '\\begin{align}\\begin{split}\n    \\hat{\\Omega} = \\down{i}\n\\\\ LHS &=\n    i\\left(\\dv{\\bt^{i}_{}}{'
            '\\tau} + \\bt^{i}_{}\\varepsilon\\right)\n\\\\ RHS &=\n%\n%\n(\\bar{f}\\bh^{\\blue{}\\red{z}}_{\\blue{}\\r'
            'ed{}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\'
            'bt^{\\blue{}\\red{z}}\n\\end{split}\\end{align}\n\n'
        )

        assert function_output == expected_result

    def test_wrap_align_environment_d_omega(self):
        """omega.name=='d' """

        # input data
        omega = fcc.general_operator_namedtuple(name='d', rank=1, m=1, n=0)
        rank_name = 'LINEAR'
        lhs = 'i\\left(\\dv{\\bt^{}_{i}}{\\tau} + \\bt^{}_{i}\\varepsilon\\right)'
        eqns = str(
            '(f\\bh^{\\blue{}\\red{}}_{\\blue{}\\red{z}})\n%\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected te'
            'rms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt_{\\blue{}\\red{z}}'
        )

        # run function
        function_output = fcc._wrap_align_environment(omega, rank_name, lhs, eqns)
        expected_result = str(
            '\\begin{align}\\begin{split}\n    \\hat{\\Omega} = \\up{i}\n\\\\ LHS &=\n    i\\left(\\dv{\\bt^{}_{i}}{\\t'
            'au} + \\bt^{}_{i}\\varepsilon\\right)\n\\\\ RHS &=\n%\n%\n(f\\bh^{\\blue{}\\red{}}_{\\blue{}\\red{z}})\n%'
            '\n%\n\\\\  &+\n%\n%\n    \\textit{no linked disconnected terms}\n%\n%\n\\\\  &+\n%\n%\n(\\bh_0)\\bt_{\\blu'
            'e{}\\red{z}}\n\\end{split}\\end{align}\n\n'
        )

        assert function_output == expected_result

    def test_wrap_align_environment_blank_omega(self):
        """omega.name=='' """

        # input data
        omega = fcc.general_operator_namedtuple(name='', rank=0, m=0, n=0)
        rank_name = '0 order'
        lhs = 'i\\left(\\varepsilon\\right)'
        eqns = str(
            '(\\bh_0 + \\bh^{\\blue{}\\red{}}_{\\blue{i}\\red{}}\\bt^{\\blue{i}\\red{}} + \\bh^{\\blue{i}\\red{}}_{\\bl'
            'ue{}\\red{}}\\bt_{\\blue{i}\\red{}}) + () + ()'
        )

        # run function
        function_output = fcc._wrap_align_environment(omega, rank_name, lhs, eqns)
        expected_result = str(
            '\\begin{align}\\begin{split}\n    \\hat{\\Omega} = 1\n\\\\ LHS &=\n    i\\left(\\varepsilon\\right)\n\\\\ '
            'RHS &=\n%\n%\n(\\bh_0 + \\bh^{\\blue{}\\red{}}_{\\blue{i}\\red{}}\\bt^{\\blue{i}\\red{}} + \\bh^{\\blue{i}'
            '\\red{}}_{\\blue{}\\red{}}\\bt_{\\blue{i}\\red{}}) + () + ()\n\\end{split}\\end{align}\n\n'
        )

        assert function_output == expected_result


class Test_main_fcc_latex_func:
    # runs main function for coverage purposes
    # TODO file compare test
    def test_generate_full_cc_latex(self, tmpdir):
        """ x """
        output_path = join(tmpdir, "latex_excited_Test_main_fcc_latex_func.tex")
        fcc.generate_full_cc_latex([2, 2, 2, 2], only_ground_state=False, path=output_path)

        output_path = join(tmpdir, "latex_ground_Test_main_fcc_latex_func.tex")
        fcc.generate_full_cc_latex([2, 2, 2, 2], only_ground_state=True, path=output_path)
