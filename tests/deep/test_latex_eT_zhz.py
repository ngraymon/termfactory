# system imports
# import re
# import pytest

# local imports
from . import context
from . import test_vars as vars
import latex_eT_zhz as et
import latex_full_cc as fcc
import latex_zhz as zhz
import namedtuple_defines as nt


# global vars
none_gen_op_nt = et.general_operator_namedtuple(name=None, rank=0, m=0, n=0)
zero_gen_op_nt = et.general_operator_namedtuple(name='', rank=0, m=0, n=0)
zero_h_op_nt = fcc.h_operator_namedtuple(rank=0, m=0, n=0)
zero_disconnected_z_right = et.disconnected_eT_z_right_operator_namedtuple(
    rank=0,
    m=0, n=0,
    m_lhs=0, n_lhs=0,
    m_t=(0,), n_t=(0,),
    m_h=0, n_h=0,
    m_l=0, n_l=0
)
zero_disconnected_t_op_nt = et.disconnected_t_operator_namedtuple(
    rank=0,
    m=0, n=0,
    m_lhs=0, n_lhs=0,
    m_l=0, n_l=0,
    m_h=0, n_h=0,
    m_r=0, n_r=0
)
zero_con_lhs_op_nt = et.connected_eT_lhs_operator_namedtuple(
    rank=0,
    m=0, n=0,
    m_l=0, n_l=0,
    m_t=[0], n_t=[0],
    m_h=0, n_h=0,
    m_r=0, n_r=0
)
zero_con_h_op_nt = et.connected_eT_h_z_operator_namedtuple(
    rank=0,
    m=0, n=0,
    m_lhs=0, n_lhs=0,
    m_t=[0], n_t=[0],
    m_l=0, n_l=0,
    m_r=0, n_r=0
)
blank_offset_dict = {
    'lower_h': '',
    'upper_h': '',
    'lower_left_z': '',
    'upper_left_z': '',
    'lower_right_z': '',
    'upper_right_z': '',
    'unlinked_index': 0,
    'summation_index': 0
}

# Builder Functions


# ----------------------------------------------------------------------------------------------- #
# -----------------------------------  BUILDER FUNCTIONS  --------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# all operators can contract with the LeftHandSide(lhs) and Hamiltonian(h)
_key_list = ['rank', 'm', 'n', 'm_lhs', 'n_lhs', 'm_h', 'n_h', ]

# t operators also contract with the left and right Z operators
t_key_list = _key_list + ['m_l', 'n_l', 'm_r', 'n_r']

# Z operators also contract with t operators and each other
zL_key_list = _key_list + ['m_t', 'n_t', 'm_r', 'n_r']
zR_key_list = _key_list + ['m_t', 'n_t', 'm_l', 'n_l']


def _make_t_default_dict():
    return dict.fromkeys(t_key_list, 0)

def _make_zL_default_dict():
    d = dict.fromkeys(zL_key_list, 0)
    d['m_t'] = d['n_t'] = (0, )  # the t contractions are tuples of zeros
    return d

def _make_zR_default_dict():
    d = dict.fromkeys(zR_key_list, 0)
    d['m_t'] = d['n_t'] = (0, )  # the t contractions are tuples of zeros
    return d

t_default_dict = _make_t_default_dict()
zL_default_dict = _make_zL_default_dict()
zR_default_dict = _make_zR_default_dict()

# -------------------------  verify FUNCTIONS  ------------------------------ #

def _verify_keys(kwargs, verification_dictionary):
    """ Returns `False` if any keys in `kwargs` are not in `verification_dictionary`."""
    for key, value in kwargs.items():
        if key not in verification_dictionary:
            # print(f"Key {key} not present in {verification_dictionary}")
            return False
    else:
        return True

def _verify_t_keys(**kwargs):
    """ Returns `False` if any keys in `kwargs` are not present in `t_default_dict`.
    It is fine if not ALL of the keys are present.
    """
    return _verify_keys(kwargs, t_default_dict)

def _verify_zL_keys(**kwargs):
    """ Returns `False` if any keys in `kwargs` are not present in `zL_default_dict`.
    It is fine if not ALL of the keys are present.
    """
    return _verify_keys(kwargs, zL_default_dict)

def _verify_zR_keys(**kwargs):
    """ Returns `False` if any keys in `kwargs` are not present in `zR_default_dict`.
    It is fine if not ALL of the keys are present.
    """
    return _verify_keys(kwargs, zR_default_dict)

# -------------------------  consistency FUNCTIONS  ------------------------------ #

def _basic_t_consistency(d):
    """ Just make sure the most basic rules are followed.
    1: `rank` = `m` + `n`
    2: `m` = sum(all `m_` terms)
    2: `n` = sum(all `n_` terms)
    """
    m_terms = ['m_lhs', 'm_l', 'm_h', 'm_r', ]
    n_terms = ['n_lhs', 'n_l', 'n_h', 'n_r', ]

    b1 = bool(d['rank'] == d['m'] + d['n'])
    b2 = bool(d['m'] == sum([d[s] for s in m_terms]))
    b3 = bool(d['n'] == sum([d[s] for s in n_terms]))

    if b1 and b2 and b3:
        return True

    string = (
        'Invalid operator\n'
        f"{b1} {d['rank'] = } == {d['m'] = } + {d['n'] = }\n"
        f"{b2} {d['m'] = } == {d['m_lhs'] = } + {d['m_l'] = } + {d['m_h'] = } + {d['m_r'] = }\n"
        f"{b3} {d['n'] = } == {d['n_lhs'] = } + {d['n_l'] = } + {d['n_h'] = } + {d['n_r'] = }\n"
    )
    raise Exception(string)

def _basic_zL_consistency(d):
    """ Just make sure the most basic rules are followed.
    1: `rank` = `m` + `n`
    2: `m` = sum(all `m_` terms)
    2: `n` = sum(all `n_` terms)
    """
    m_terms = ['m_lhs', 'm_h', 'm_r', ]
    n_terms = ['n_lhs', 'n_h', 'n_r', ]

    b1 = bool(d['rank'] == d['m'] + d['n'])
    b2 = bool(d['m'] == sum([d[s] for s in m_terms]) + sum(d['m_t']))
    b3 = bool(d['n'] == sum([d[s] for s in n_terms]) + sum(d['n_t']))

    if b1 and b2 and b3:
        return True

    string = (
        'Invalid operator\n'
        f"{b1} {d['rank'] = } == {d['m'] = } + {d['n'] = }\n"
        f"{b2} {d['m'] = } == {d['m_lhs'] = } + sum({d['m_t'] = }) + {d['m_h'] = } + {d['m_r'] = }\n"
        f"{b3} {d['n'] = } == {d['n_lhs'] = } + sum({d['n_t'] = }) + {d['n_h'] = } + {d['n_r'] = }\n"
    )
    raise Exception(string)

def _basic_zR_consistency(d):
    """ Just make sure the most basic rules are followed.
    1: `rank` = `m` + `n`
    2: `m` = sum(all `m_` terms)
    2: `n` = sum(all `n_` terms)
    """
    m_terms = ['m_lhs', 'm_h', 'm_l', ]
    n_terms = ['n_lhs', 'n_h', 'n_l', ]

    b1 = bool(d['rank'] == d['m'] + d['n'])
    b2 = bool(d['m'] == sum([d[s] for s in m_terms]) + sum(d['m_t']))
    b3 = bool(d['n'] == sum([d[s] for s in n_terms]) + sum(d['n_t']))

    if b1 and b2 and b3:
        return True

    string = (
        'Invalid operator\n'
        f"{b1} {d['rank'] = } == {d['m'] = } + {d['n'] = }\n"
        f"{b2} {d['m'] = } == {d['m_lhs'] = } + sum({d['m_t'] = }) + {d['m_h'] = } + {d['m_l'] = }\n"
        f"{b3} {d['n'] = } == {d['n_lhs'] = } + sum({d['n_t'] = }) + {d['n_h'] = } + {d['n_l'] = }\n"
    )
    raise Exception(string)

# --------------------  operator construction FUNCTIONS  ------------------------- #

def build_t_operator(status, **kwargs):
    """ Return a namedtuple specified by a dictionary.
    Input argument `kwargs` takes precedence over `default_dict`.
    """
    assert _verify_t_keys(kwargs), f'invalid kwargs: {kwargs}'
    dictionary = _make_t_default_dict().update(**kwargs)
    assert _basic_t_consistency(dictionary), f'invalid operator {dictionary}'

    # build the operator
    if status == "connected":
        built_tuple = et.connected_t_operator_namedtuple(**dictionary)
    elif status == "disconnected":
        built_tuple = et.disconnected_t_operator_namedtuple(**dictionary)
    else:
        raise Exception(
            f"Status {status} is not valid, should be 'connected' or 'disconnected'\n"
            f"Input dictionary:\n{kwargs}"
        )

    return built_tuple

def build_zL_operator_namedtuple(status, **kwargs):
    """ Wrapper function for `_build_eT_z_op`
    """
    assert _verify_zL_keys(kwargs), f'invalid kwargs: {kwargs}'
    dictionary = _make_zL_default_dict().update(**kwargs)
    assert _basic_zL_consistency(dictionary), f'invalid operator {dictionary}'

    # build the operator
    if status == "connected":
        built_tuple = et.connected_eT_z_right_operator_namedtuple(**dictionary)
    elif status == "disconnected":
        built_tuple = et.disconnected_eT_z_right_operator_namedtuple(**dictionary)
    else:
        raise Exception("status not connected/disconnected, typo?")

    return built_tuple

    return

def build_zR_operator_namedtuple(status, **kwargs):
    """ Wrapper function for `_build_eT_z_op`
    """
    assert _verify_zR_keys(kwargs), f'invalid kwargs: {kwargs}'
    dictionary = _make_zR_default_dict().update(**kwargs)
    assert _basic_zR_consistency(dictionary), f'invalid operator {dictionary}'

    # build the operator
    if status == "connected":
        built_tuple = et.connected_eT_z_right_operator_namedtuple(**dictionary)
    elif status == "disconnected":
        built_tuple = et.disconnected_eT_z_right_operator_namedtuple(**dictionary)
    else:
        raise Exception("status not connected/disconnected, typo?")

    return built_tuple

def build_z_operator_namedtuple(side, status, **kwargs):
    """Wrapper for `build_zL_operator_namedtuple` and
    `build_zR_operator_namedtuple`.
    """
    if side == 'left':
        build_zL_operator_namedtuple(status, kwargs)
    elif side == 'right':
        build_zR_operator_namedtuple(status, kwargs)
    else:
        raise Exception(f"Incorrect {side = }, should be either 'left' or 'right'.")

# -------------------------  Z FUNCTIONS  ------------------------------ #

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

class Test_generate_eT_operator:

    def test_basic(self):
        maximum_eT_rank = 2
        function_output = et.generate_eT_operator(maximum_eT_rank)
        expected_result = et.eT_operator_namedtuple(
            maximum_rank=2,
            operator_list=[
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ]
        )
        assert function_output == expected_result


class Test_generate_eT_taylor_expansion:

    def test_basic(self):
        maximum_eT_rank = 2
        eT_taylor_max_order = 3
        function_output = et.generate_eT_taylor_expansion(maximum_eT_rank, eT_taylor_max_order)
        expected_result = vars.generate_eT_taylor_expansion.expansion
        assert function_output == expected_result


class Test_z_joining_with_z_terms_eT:

    def test_basic(self):
        LHS = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        left_z = none_gen_op_nt
        right_z = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._z_joining_with_z_terms_eT(LHS, t_list, h, left_z, right_z)
        expected_result = False
        assert function_output == expected_result

    def test_req_case_1(self):
        """(required_b_for_left_z > available_b_for_left_z)"""
        LHS = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        left_z = et.general_operator_namedtuple(name=None, rank=0, m=1, n=0)
        right_z = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._z_joining_with_z_terms_eT(LHS, t_list, h, left_z, right_z)
        expected_result = True
        assert function_output == expected_result

    def test_req_case_2(self):
        """(required_d_for_left_z > available_d_for_left_z)"""
        LHS = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        left_z = et.general_operator_namedtuple(name=None, rank=0, m=0, n=1)
        right_z = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._z_joining_with_z_terms_eT(LHS, t_list, h, left_z, right_z)
        expected_result = True
        assert function_output == expected_result

    def test_req_case_3(self):
        """(required_b_for_right_z > available_b_for_right_z)"""
        LHS = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        left_z = none_gen_op_nt
        right_z = et.general_operator_namedtuple(name='z', rank=0, m=1, n=0)
        function_output = et._z_joining_with_z_terms_eT(LHS, t_list, h, left_z, right_z)
        expected_result = True
        assert function_output == expected_result

    def test_req_case_4(self):
        """(required_d_for_right_z > available_d_for_right_z)"""
        LHS = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        left_z = none_gen_op_nt
        right_z = et.general_operator_namedtuple(name='z', rank=0, m=0, n=1)
        function_output = et._z_joining_with_z_terms_eT(LHS, t_list, h, left_z, right_z)
        expected_result = True
        assert function_output == expected_result


class Test_t_joining_with_t_terms_eT:

    def test_false_case(self):
        omega = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._t_joining_with_t_terms_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_true_case(self):
        """(required_b > available_b) or (required_d > available_d)"""
        omega = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=1, n=0)]
        h = zero_h_op_nt
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._t_joining_with_t_terms_eT(omega, t_list, h, z_left, z_right)
        expected_result = True
        assert function_output == expected_result


class Test_omega_joining_with_itself_eT:

    def test_omega_zero_case(self):
        """(omega.m == 0) or (omega.n == 0)"""
        omega = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._omega_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_omega_n_non_zero_false_case_1(self):
        """omega.n > 0 and (h.m > 0) or (z_left.m > 0) or (z_right.m > 0)"""
        omega = et.general_operator_namedtuple(name='', rank=1, m=1, n=1)
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = fcc.h_operator_namedtuple(rank=0, m=1, n=0)
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._omega_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_omega_n_non_zero_false_case_2(self):
        """omega.n > 0 and if t.m > 0"""
        omega = et.general_operator_namedtuple(name='', rank=1, m=1, n=1)
        t_list = [et.general_operator_namedtuple(name='1', rank=1, m=1, n=1)]
        h = zero_h_op_nt
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._omega_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_omega_m_non_zero_false_case_1(self):
        omega = et.general_operator_namedtuple(name='', rank=1, m=1, n=1)
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = fcc.h_operator_namedtuple(rank=0, m=0, n=1)
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._omega_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_omega_m_non_zero_false_case_2(self):
        omega = et.general_operator_namedtuple(name='', rank=1, m=1, n=1)
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=1)]
        h = zero_h_op_nt
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._omega_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_true_case(self):
        omega = et.general_operator_namedtuple(name='', rank=1, m=1, n=1)
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._omega_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = True
        assert function_output == expected_result


class Test_h_joining_with_itself_eT:

    def test_zero_case(self):
        omega = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = zero_h_op_nt
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._h_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_h_n_nonzero_false_case_1(self):
        omega = et.general_operator_namedtuple(name='', rank=0, m=1, n=0)
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = fcc.h_operator_namedtuple(rank=1, m=1, n=1)
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._h_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_h_n_nonzero_false_case_2(self):
        omega = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=1, n=0)]
        h = fcc.h_operator_namedtuple(rank=1, m=1, n=1)
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._h_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_h_m_nonzero_false_case_1(self):
        omega = et.general_operator_namedtuple(name='', rank=0, m=0, n=1)
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = fcc.h_operator_namedtuple(rank=1, m=1, n=1)
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._h_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_h_m_nonzero_false_case_2(self):
        omega = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=1)]
        h = fcc.h_operator_namedtuple(rank=1, m=1, n=1)
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._h_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = False
        assert function_output == expected_result

    def test_true_case(self):
        omega = zero_gen_op_nt
        t_list = [et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]
        h = fcc.h_operator_namedtuple(rank=1, m=1, n=1)
        z_left = none_gen_op_nt
        z_right = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._h_joining_with_itself_eT(omega, t_list, h, z_left, z_right)
        expected_result = True
        assert function_output == expected_result


class Test_generate_valid_eT_z_n_operator_permutations:

    def test_basic(self):
        LHS = zero_gen_op_nt
        eT = [[et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]]
        h = zero_h_op_nt
        all_z_permutations = [
            (
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
            ),
            (
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            )
        ]
        function_output = et._generate_valid_eT_z_n_operator_permutations(LHS, eT, h, all_z_permutations)
        expected_result = [
            (
                (
                    et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),
                ),
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
            )
        ]
        assert function_output == expected_result

    def test_perm_T_greater_than_one(self):
        """len(perm_T) != 1"""
        LHS = zero_gen_op_nt
        eT = [
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ],
            [
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2),
                et.general_operator_namedtuple(name='s_2', rank=2, m=0, n=2)
            ]
        ]
        h = zero_h_op_nt
        all_z_permutations = [
            (
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
            ),
            (
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ),
            (
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z^2', rank=2, m=2, n=0)
            )
        ]
        function_output = et._generate_valid_eT_z_n_operator_permutations(LHS, eT, h, all_z_permutations)
        expected_result = [
            (
                (
                    et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                    et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
                ),
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z^2', rank=2, m=2, n=0)
            )
        ]
        assert function_output == expected_result


class Test_generate_all_valid_eT_z_connection_permutations:

    def test_basic(self):
        LHS = zero_gen_op_nt
        t_list = (et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),)
        h = zero_h_op_nt
        left_z = none_gen_op_nt
        right_z = et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
        function_output = et._generate_all_valid_eT_z_connection_permutations(
            LHS, t_list, h, left_z, right_z, log_invalid=True)
        expected_result = ([((0, (0,), 0, 0), (0, (0,), 0, 0))], [((0, (0,), 0, 0), (0, (0,), 0, 0))])  # TODO expand?
        assert function_output == expected_result


class Test_generate_all_valid_eT_connection_permutations:

    def test_basic(self):
        LHS = zero_gen_op_nt
        t_list = (et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),)
        h = zero_h_op_nt
        z_pair = (
            None,
            zero_disconnected_z_right
        )
        function_output = et._generate_all_valid_eT_connection_permutations(LHS, t_list, h, z_pair, log_invalid=True)
        expected_result = ([[[0, 0, 0, 0]]], [[[0, 0, 0, 0]]])
        assert function_output == expected_result

    def test_z_right_is_none(self):
        LHS = zero_gen_op_nt
        t_list = (et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),)
        h = zero_h_op_nt
        z_pair = (
            et.disconnected_eT_z_left_operator_namedtuple(
                rank=0,
                m=0, n=0,
                m_lhs=0, n_lhs=0,
                m_h=0, n_h=0,
                m_r=0, n_r=0,
                m_t=(0,), n_t=(0,)
            ),
            None
        )
        function_output = et._generate_all_valid_eT_connection_permutations(LHS, t_list, h, z_pair, log_invalid=True)
        expected_result = ([[[0, 0, 0, 0]]], [[[0, 0, 0, 0]]])
        assert function_output == expected_result

    def test_rem_m_zero(self):
        """elif remaining_m == 0:"""
        LHS = zero_gen_op_nt
        t_list = (et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),)
        h = fcc.h_operator_namedtuple(rank=1, m=1, n=0)
        z_pair = (
            None,
            et.disconnected_eT_z_right_operator_namedtuple(
                rank=0,
                m=0, n=0,
                m_lhs=0, n_lhs=0,
                m_t=(0,), n_t=(0,),
                m_h=0, n_h=0,
                m_l=0, n_l=0
            )
        )
        function_output = et._generate_all_valid_eT_connection_permutations(LHS, t_list, h, z_pair, log_invalid=True)
        expected_result = ([((0, 0, 0, 0),)], [((0, 0, 1, 0),)])
        assert function_output == expected_result


class Test_generate_all_o_eT_h_z_connection_permutations:

    def test_basic(self):
        LHS = zero_gen_op_nt
        h = zero_h_op_nt
        valid_permutations = [
            (
                (
                    et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),
                ),
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
            )
        ]
        function_output = et._generate_all_o_eT_h_z_connection_permutations(
            LHS, h, valid_permutations, found_it_bool=False)
        expected_result = [
            (
                (
                    zero_disconnected_t_op_nt,
                ),
                (
                    None,
                    zero_disconnected_z_right
                )
            )
        ]
        assert function_output == expected_result

    def test_connected_z(self):
        """if the Z operator is connected (at least 1 connection to H)"""
        LHS = zero_gen_op_nt
        h = fcc.h_operator_namedtuple(rank=1, m=0, n=1)
        valid_permutations = [
            (
                (
                    et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),
                ),
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            )
        ]
        function_output = et._generate_all_o_eT_h_z_connection_permutations(
            LHS, h, valid_permutations, found_it_bool=False
        )
        expected_result = [
            (
                (
                    zero_disconnected_t_op_nt,
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(0,), n_t=(0,),
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            )
        ]
        assert function_output == expected_result

    def test_t_list_long(self):
        """if len(t_list) != 1 or t_list[0].rank != 0:"""
        LHS = et.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        h = fcc.h_operator_namedtuple(rank=1, m=1, n=0)
        valid_permutations = [
            (
                (
                    et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1),
                ),
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            )
        ]
        function_output = et._generate_all_o_eT_h_z_connection_permutations(
            LHS, h, valid_permutations, found_it_bool=False
        )
        expected_result = [
            (
                (
                    build_t_operator("disconnected", rank=1, n=1, n_r=1),
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(1,), n_t=(0,),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=1, n_lhs=0,
                        m_t=(0,), n_t=(0,),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            )
        ]
        assert function_output == expected_result


class Test_remove_duplicate_t_tuple_permutations:

    def test_basic(self):
        LHS = zero_gen_op_nt
        h = zero_h_op_nt
        eT_connection_permutations = [
            (
                (
                    zero_disconnected_t_op_nt,
                ),
                (
                    None,
                    zero_disconnected_z_right))
        ]
        function_output = et._remove_duplicate_t_tuple_permutations(LHS, h, eT_connection_permutations)
        expected_result = (
            [
                (
                    (zero_disconnected_t_op_nt,),
                    (None, zero_disconnected_z_right)
                )
            ],
            {
                (
                    (zero_disconnected_t_op_nt,),
                    (None, zero_disconnected_z_right)
                ): 1
            }
        )
        assert function_output == expected_result

    def test_long_tuple_case_1(self):
        LHS = et.general_operator_namedtuple(name='bb', rank=2, m=0, n=2)
        h = fcc.h_operator_namedtuple(rank=2, m=2, n=0)
        eT_connection_permutations = [  # file flag
            (
                (
                    et.disconnected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=1
                    ),
                    et.disconnected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=1
                    )
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=2,
                        m=2, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(1, 1), n_t=(0, 0),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                    et.disconnected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=1
                    )
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=2,
                        m=2, n=0,
                        m_lhs=1, n_lhs=0,
                        m_t=(0, 1), n_t=(0, 0),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.disconnected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=1
                    ),
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    )
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=2,
                        m=2, n=0,
                        m_lhs=1, n_lhs=0,
                        m_t=(1, 0), n_t=(0, 0),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    )
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=2,
                        m=2, n=0,
                        m_lhs=2, n_lhs=0,
                        m_t=(0, 0), n_t=(0, 0),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            )
        ]
        function_output = et._remove_duplicate_t_tuple_permutations(LHS, h, eT_connection_permutations)
        expected_result = (  # file flag
            [
                (
                    (
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        ),
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=0, n_lhs=0,
                            m_t=(1, 1), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                ),
                (
                    (
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        ),
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=1, n_lhs=0,
                            m_t=(0, 1), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                ),
                (
                    (
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        ),
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=1, n_lhs=0,
                            m_t=(1, 0), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                ),
                (
                    (
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        ),
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=2, n_lhs=0,
                            m_t=(0, 0), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                )
            ],
            {
                (
                    (
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        ),
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=0, n_lhs=0,
                            m_t=(1, 1), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                ): 1,
                (
                    (
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        ),
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=1, n_lhs=0,
                            m_t=(0, 1), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                ): 1,
                (
                    (
                        et.disconnected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=0,
                            m_r=0, n_r=1
                        ),
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=1, n_lhs=0,
                            m_t=(1, 0), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                ): 1,
                (
                    (
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        ),
                        et.connected_t_operator_namedtuple(
                            rank=1,
                            m=0, n=1,
                            m_lhs=0, n_lhs=0,
                            m_l=0, n_l=0,
                            m_h=0, n_h=1,
                            m_r=0, n_r=0
                        )
                    ),
                    (
                        None,
                        et.disconnected_eT_z_right_operator_namedtuple(
                            rank=2,
                            m=2, n=0,
                            m_lhs=2, n_lhs=0,
                            m_t=(0, 0), n_t=(0, 0),
                            m_h=0, n_h=0,
                            m_l=0, n_l=0
                        )
                    )
                ): 1
            }
        )
        assert function_output == expected_result


class Test_generate_explicit_eT_z_connections:

    def test_basic(self):
        LHS = zero_gen_op_nt
        h = zero_h_op_nt
        unique_permutations = [
            ((zero_disconnected_t_op_nt,),
                (None, zero_disconnected_z_right))
        ]
        prefactor_count = dict([(perm, 1) for perm in unique_permutations])  # this might be sufficient for now
        function_output = et._generate_explicit_eT_z_connections(LHS, h, unique_permutations, prefactor_count)
        expected_result = [
            [
                zero_con_lhs_op_nt,
                (zero_disconnected_t_op_nt,),
                et.connected_eT_h_z_operator_namedtuple(
                    rank=0,
                    m=0, n=0,
                    m_lhs=0, n_lhs=0,
                    m_t=[0], n_t=[0],
                    m_l=0, n_l=0,
                    m_r=0, n_r=0
                ),
                (
                    None,
                    zero_disconnected_z_right
                ),
                1
            ]
        ]
        assert function_output == expected_result

    def test_unequal_kwargs_case_1(self):
        """if h_kwargs['m_lhs'] != lhs_kwargs['n_h']: """
        LHS = zero_gen_op_nt
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        unique_permutations = [
            (
                (
                    et.disconnected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=1
                    ),
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(1,), n_t=(0,),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(0,), n_t=(0,),
                        m_h=1, n_h=0, m_l=0,
                        n_l=0
                    )
                )
            )
        ]
        prefactor_count = dict([(perm, 1) for perm in unique_permutations])  # this might be sufficient for now
        function_output = et._generate_explicit_eT_z_connections(LHS, h, unique_permutations, prefactor_count)
        expected_result = [
            [
                zero_con_lhs_op_nt,
                (
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                ),
                et.connected_eT_h_z_operator_namedtuple(
                    rank=2,
                    m=1, n=1,
                    m_lhs=0, n_lhs=0,
                    m_t=[1], n_t=[0],
                    m_l=0, n_l=0,
                    m_r=0, n_r=1
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(0,), n_t=(0,),
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                ),
                1
            ]
        ]
        assert function_output == expected_result

    def test_unequal_kwargs_case_2(self):
        """elif h_kwargs['n_lhs'] != lhs_kwargs['m_h']:"""
        LHS = zero_gen_op_nt
        h = fcc.h_operator_namedtuple(rank=2, m=1, n=1)
        unique_permutations = [  # file flag
            (
                (
                    et.disconnected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=1
                    ),
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(1,), n_t=(0,),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(0,), n_t=(0,),
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=2,
                        m=0, n=2,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=1
                    ),
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(1,), n_t=(0,),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=2,
                        m=0, n=2,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=1
                    ),
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(0,), n_t=(0,),
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.disconnected_t_operator_namedtuple(
                        rank=2,
                        m=0, n=2,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=0,
                        m_r=0, n_r=2
                    ),
                ),
                (
                    None,
                    et.disconnected_eT_z_right_operator_namedtuple(
                        rank=2,
                        m=2, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(2,), n_t=(0,),
                        m_h=0, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            ),
            (
                (
                    et.connected_t_operator_namedtuple(
                        rank=2,
                        m=0, n=2,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=1
                    ),
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=2,
                        m=2, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(1,), n_t=(0,),
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                )
            )
        ]
        prefactor_count = dict([(perm, 1) for perm in unique_permutations])  # this might be sufficient for now
        function_output = et._generate_explicit_eT_z_connections(LHS, h, unique_permutations, prefactor_count)
        expected_result = [
            [
                zero_con_lhs_op_nt,
                (
                    et.connected_t_operator_namedtuple(
                        rank=1,
                        m=0, n=1,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=0
                    ),
                ),
                et.connected_eT_h_z_operator_namedtuple(
                    rank=2,
                    m=1, n=1,
                    m_lhs=0, n_lhs=0,
                    m_t=[1], n_t=[0],
                    m_l=0, n_l=0,
                    m_r=0, n_r=1
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=1,
                        m=1, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(0,), n_t=(0,),
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                ), 1
            ],
            [
                zero_con_lhs_op_nt,
                (
                    et.connected_t_operator_namedtuple(
                        rank=2,
                        m=0, n=2,
                        m_lhs=0, n_lhs=0,
                        m_l=0, n_l=0,
                        m_h=0, n_h=1,
                        m_r=0, n_r=1
                    ),
                ),
                et.connected_eT_h_z_operator_namedtuple(
                    rank=2,
                    m=1, n=1,
                    m_lhs=0, n_lhs=0,
                    m_t=[1], n_t=[0],
                    m_l=0, n_l=0,
                    m_r=0, n_r=1
                ),
                (
                    None,
                    et.connected_eT_z_right_operator_namedtuple(
                        rank=2,
                        m=2, n=0,
                        m_lhs=0, n_lhs=0,
                        m_t=(1,), n_t=(0,),
                        m_h=1, n_h=0,
                        m_l=0, n_l=0
                    )
                ),
                1
            ]
        ]
        assert function_output == expected_result


class Test_build_eThz_latex_prefactor:

    def test_basic(self):
        t_list = (zero_disconnected_t_op_nt,)
        h = zero_con_h_op_nt
        z_left = None
        z_right = zero_disconnected_z_right
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = ''
        assert function_output == expected_result

    def test_f_factor(self):
        t_list = (
            et.disconnected_t_operator_namedtuple(
                rank=1,
                m=0, n=1,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=0,
                m_r=0, n_r=1
            ),
            et.disconnected_t_operator_namedtuple(
                rank=1,
                m=0, n=1,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=0,
                m_r=0, n_r=1
            )
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=0,
            m=0, n=0,
            m_lhs=0, n_lhs=0,
            m_t=[0, 0], n_t=[0, 0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        z_left = None
        z_right = et.disconnected_eT_z_right_operator_namedtuple(
            rank=2,
            m=2, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(1, 1), n_t=(0, 0),
            m_h=0, n_h=0,
            m_l=0, n_l=0
        )
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{2!}{2!2!}'
        assert function_output == expected_result

    def test_n_factor(self):
        t_list = (zero_disconnected_t_op_nt,)
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=2,
            m=0, n=2,
            m_lhs=0, n_lhs=0,
            m_t=[0], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=2
        )
        z_left = None
        z_right = et.connected_eT_z_right_operator_namedtuple(
            rank=2,
            m=2, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(0,), n_t=(0,),
            m_h=2, n_h=0,
            m_l=0, n_l=0
        )
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{2!}{2!2!}'
        assert function_output == expected_result

    def test_choose_result(self):
        """if choose_result > 1:"""
        t_list = (
            et.connected_t_operator_namedtuple(
                rank=2,
                m=0, n=2,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=1,
                m_r=0, n_r=1
            ),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_t=[1], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        z_left = None
        z_right = et.disconnected_eT_z_right_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(1,), n_t=(0,),
            m_h=0, n_h=0,
            m_l=0, n_l=0
        )
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{1}{2!}'
        assert function_output == expected_result

    def test_internal_perms(self):
        """if internal_perms > 1:"""
        t_list = (et.connected_t_operator_namedtuple(
            rank=2,
            m=0, n=2,
            m_lhs=0, n_lhs=0,
            m_l=0, n_l=0,
            m_h=0, n_h=2,
            m_r=0, n_r=0),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=2,
            m=3, n=0,
            m_lhs=1, n_lhs=0,
            m_t=[2], n_t=[0],
            m_l=0, n_l=0,
            m_r=1, n_r=0
        )
        z_left = None
        z_right = et.disconnected_eT_z_right_operator_namedtuple(
            rank=0,
            m=0, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(0,), n_t=(0,),
            m_h=0, n_h=0,
            m_l=0, n_l=0
        )
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{(3)(2)2!}{3!2!}'
        assert function_output == expected_result

    def test_h_m_t_count(self):
        """count_t = sum(h.m_t)
           if count_t > 1:"""
        t_list = (et.connected_t_operator_namedtuple(
            rank=2,
            m=0, n=2,
            m_lhs=0, n_lhs=0,
            m_l=0, n_l=0,
            m_h=0, n_h=2,
            m_r=0, n_r=0),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=2,
            m=2, n=0,
            m_lhs=0, n_lhs=0,
            m_t=[2], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        z_left = None
        z_right = build_eT_z_operator_namedtuple('disconnected', 'right')
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{2!}{2!2!}'
        assert function_output == expected_result

    def test_external_perms(self):
        """
        external_perms = math.comb(h.n, h.n_lhs)
        if external_perms > 1:
        """
        t_list = (et.connected_t_operator_namedtuple(
            rank=2,
            m=0, n=2,
            m_lhs=0, n_lhs=0,
            m_l=0, n_l=0,
            m_h=0, n_h=2,
            m_r=0, n_r=0),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=2,
            m=2, n=3,
            m_lhs=0, n_lhs=2,
            m_t=[2], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        z_left = None
        z_right = et.disconnected_eT_z_right_operator_namedtuple(
            rank=0,
            m=0, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(0,), n_t=(0,),
            m_h=0, n_h=0,
            m_l=0, n_l=0
        )
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{2!(3)}{2!3!2!}'
        assert function_output == expected_result

    def test_h_n_internal_perms(self):
        """
        new_max = h.n - h.n_lhs
        internal_perms = math.comb(new_max, h.n_r)
        if internal_perms > 1:
        """
        t_list = (et.connected_t_operator_namedtuple(
            rank=2,
            m=0, n=2,
            m_lhs=0, n_lhs=0,
            m_l=0, n_l=0,
            m_h=0, n_h=2,
            m_r=0, n_r=0),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=2,
            m=2, n=4,
            m_lhs=0, n_lhs=1,
            m_t=[2], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=2
        )
        z_left = None
        z_right = et.disconnected_eT_z_right_operator_namedtuple(
            rank=0,
            m=0, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(0,), n_t=(0,),
            m_h=0, n_h=0,
            m_l=0, n_l=0
        )
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{2!(4)(3)}{2!4!2!}'
        assert function_output == expected_result

    def test_account_for_interal_perms(self):
        """account for the permutations of eT-Z internal labels"""
        t_list = (et.connected_t_operator_namedtuple(
            rank=2,
            m=0, n=2,
            m_lhs=0, n_lhs=0,
            m_l=0, n_l=0,
            m_h=0, n_h=2,
            m_r=0, n_r=0),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=2,
            m=2, n=2,
            m_lhs=0, n_lhs=1,
            m_t=[2], n_t=[1, 2],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        z_left = None
        z_right = et.disconnected_eT_z_right_operator_namedtuple(
            rank=0,
            m=0, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(0,), n_t=(0,),
            m_h=0, n_h=0,
            m_l=0, n_l=0
        )
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{2!(2)3!}{2!2!2!}'
        assert function_output == expected_result

    def test_taylor_t_m(self):
        """if t.m > 1:"""
        t_list = (
            et.disconnected_t_operator_namedtuple(
                rank=1,
                m=2, n=1,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=0,
                m_r=0, n_r=1
            ),
            et.disconnected_t_operator_namedtuple(
                rank=1,
                m=0, n=1,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=0,
                m_r=0, n_r=1
            )
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=0,
            m=0, n=0,
            m_lhs=0, n_lhs=0,
            m_t=[0, 0], n_t=[0, 0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        z_left = None
        z_right = build_eT_z_operator_namedtuple('disconnected', 'right', rank=2, m=2, m_t=(1, 1), n_t=(0, 0))
        overcounting_prefactor = 1
        function_output = et._build_eThz_latex_prefactor(
            t_list,
            h,
            z_left,
            z_right,
            overcounting_prefactor,
            simplify_flag=True
        )
        expected_result = '\\frac{2!}{2!2!2!}'
        assert function_output == expected_result
        # TODO Test Glue cases later on / consider removing


class Test_f_t_h_contributions:

    def test_if_case(self):
        t_list = (zero_disconnected_t_op_nt,)
        h = zero_con_h_op_nt
        function_output = et._f_t_h_contributions(t_list, h)
        expected_result = [0]
        assert function_output == expected_result

    def test_else_case(self):
        t_list = (
            et.connected_t_operator_namedtuple(
                rank=1,
                m=0, n=1,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=1,
                m_r=0, n_r=0
            ),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_t=[1], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        function_output = et._f_t_h_contributions(t_list, h)
        expected_result = [0]
        assert function_output == expected_result


class Test_fbar_t_h_contributions:

    def test_if_case(self):
        t_list = (zero_disconnected_t_op_nt,)
        h = zero_con_h_op_nt
        function_output = et._fbar_t_h_contributions(t_list, h)
        expected_result = [0]
        assert function_output == expected_result

    def test_else_case(self):
        t_list = (
            et.connected_t_operator_namedtuple(
                rank=1,
                m=0, n=1,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=1,
                m_r=0, n_r=0
            ),
        )
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_t=[1], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        function_output = et._fbar_t_h_contributions(t_list, h)
        expected_result = [1]
        assert function_output == expected_result


class Test_f_t_zR_contributions:

    def test_basic_case(self):
        t_list = (zero_disconnected_t_op_nt,)
        z_right = zero_disconnected_z_right
        function_output = et._f_t_zR_contributions(t_list, z_right)
        expected_result = [0]
        assert function_output == expected_result


class Test_fbar_t_zR_contributions:

    def test_basic_case(self):
        t_list = (zero_disconnected_t_op_nt,)
        z_right = zero_disconnected_z_right
        function_output = et._fbar_t_zR_contributions(t_list, z_right)
        expected_result = [0]
        assert function_output == expected_result


class Test_build_eT_term_latex_labels:

    def test_rank_zero(self):
        t_list = (zero_disconnected_t_op_nt,)
        offset_dict = blank_offset_dict
        function_output = et._build_eT_term_latex_labels(t_list, offset_dict, color=True, letters=True)
        expected_result = '\\mathds1'
        assert function_output == expected_result

    def test_rank_non_zero_with_sub(self):
        t_list = (
            et.disconnected_t_operator_namedtuple(
                rank=1,
                m=0, n=1,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=0,
                m_r=0, n_r=1
            ),
        )
        offset_dict = blank_offset_dict
        function_output = et._build_eT_term_latex_labels(t_list, offset_dict, color=True, letters=True)
        expected_result = '\\bt^{}_{\\magenta{}\\blue{}\\magenta{k}\\red{}}'
        assert function_output == expected_result

    def test_rank_non_zero_with_sup(self):
        t_list = (
            et.disconnected_t_operator_namedtuple(  # maybe bad term
                rank=1,
                m=1, n=0,
                m_lhs=0, n_lhs=0,
                m_l=0, n_l=0,
                m_h=0, n_h=0,
                m_r=0, n_r=1
            ),
        )
        offset_dict = blank_offset_dict
        function_output = et._build_eT_term_latex_labels(t_list, offset_dict, color=True, letters=True)
        expected_result = '\\bt^{\\magenta{}\\blue{}\\magenta{}\\red{}}_{}'
        assert function_output == expected_result


class Test_build_eT_hz_term_latex_labels:

    def test_rank_zero(self):
        h = zero_con_h_op_nt
        offset_dict = blank_offset_dict
        function_output = et._build_eT_hz_term_latex_labels(h, offset_dict, color=True, letters=True)
        expected_result = '\\bh_0'
        assert function_output == expected_result

    def test_rank_non_zero_with_sub(self):
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=1,
            m=0, n=1,
            m_lhs=0, n_lhs=0,
            m_t=[0], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=1
        )
        offset_dict = blank_offset_dict
        function_output = et._build_eT_hz_term_latex_labels(h, offset_dict, color=True, letters=True)
        expected_result = '\\bh^{}_{\\blue{}\\blue{l}\\red{}}'
        assert function_output == expected_result

    def test_rank_non_zero_with_sup(self):
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_t=[1], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=0
        )
        offset_dict = {
            'lower_h': '',
            'upper_h': '\\blue{k}',
            'lower_left_z': '',
            'upper_left_z': '\\magenta{}',
            'lower_right_z': '',
            'upper_right_z': '\\magenta{}',
            'unlinked_index': 0,
            'summation_index': 1
        }
        function_output = et._build_eT_hz_term_latex_labels(h, offset_dict, color=True, letters=True)
        expected_result = '\\bh^{\\blue{k}\\blue{}\\red{}}_{}'
        assert function_output == expected_result


class Test_build_eT_right_z_term:

    def test_rank_zero(self):
        h = zero_con_h_op_nt
        z_right = zero_disconnected_z_right
        offset_dict = blank_offset_dict
        function_output = et._build_eT_right_z_term(h, z_right, offset_dict, color=True, letters=True)
        expected_result = '\\bz_0'
        assert function_output == expected_result

    def test_rank_non_zero_with_sub(self):
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=1,
            m=0, n=1,
            m_lhs=0, n_lhs=0,
            m_t=[0], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=1
        )
        z_right = et.connected_eT_z_right_operator_namedtuple(  # maybe bad z, just changed num to fid needs
            rank=1,
            m=0, n=1,
            m_lhs=0, n_lhs=0,
            m_t=(0,), n_t=(0,),
            m_h=1, n_h=0,
            m_l=0, n_l=0)
        offset_dict = {
            'lower_h': '',
            'upper_h': '',
            'lower_left_z': '',
            'upper_left_z': '',
            'lower_right_z': '',
            'upper_right_z': '\\blue{k}',
            'unlinked_index': 0,
            'summation_index': 1
        }
        function_output = et._build_eT_right_z_term(h, z_right, offset_dict, color=True, letters=True)
        expected_result = '\\bz^{}_{\\red{}}'
        assert function_output == expected_result

    def test_rank_non_zero_with_sup(self):
        h = et.connected_eT_h_z_operator_namedtuple(
            rank=1,
            m=0, n=1,
            m_lhs=0, n_lhs=0,
            m_t=[0], n_t=[0],
            m_l=0, n_l=0,
            m_r=0, n_r=1
        )
        z_right = et.connected_eT_z_right_operator_namedtuple(
            rank=1,
            m=1, n=0,
            m_lhs=0, n_lhs=0,
            m_t=(0,), n_t=(0,),
            m_h=1, n_h=0,
            m_l=0, n_l=0
        )
        offset_dict = {
            'lower_h': '',
            'upper_h': '',
            'lower_left_z': '',
            'upper_left_z': '',
            'lower_right_z': '',
            'upper_right_z': '\\blue{k}',
            'unlinked_index': 0,
            'summation_index': 1
        }
        function_output = et._build_eT_right_z_term(h, z_right, offset_dict, color=True, letters=True)
        expected_result = '\\bz^{\\blue{k}\\red{}}_{}'
        assert function_output == expected_result


class Test_prepare_third_eTz_latex:
    term_list = [  # file flag
        [
            zero_con_lhs_op_nt,
            (
                zero_disconnected_t_op_nt,
            ),
            zero_con_h_op_nt,
            (
                None,
                zero_disconnected_z_right
            ),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(
                rank=0,
                m=0, n=0,
                m_l=0, n_l=0,
                m_t=[0], n_t=[0],
                m_h=0, n_h=0,
                m_r=0, n_r=0
            ),
            (
                et.disconnected_t_operator_namedtuple(
                    rank=0,
                    m=0, n=0,
                    m_lhs=0, n_lhs=0,
                    m_l=0, n_l=0,
                    m_h=0, n_h=0,
                    m_r=0, n_r=0
                ),
            ),
            et.connected_eT_h_z_operator_namedtuple(
                rank=1,
                m=0, n=1,
                m_lhs=0, n_lhs=0,
                m_t=[0], n_t=[0],
                m_l=0, n_l=0,
                m_r=0, n_r=1
            ),
            (
                None,
                et.connected_eT_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=0, n_lhs=0,
                    m_t=(0,), n_t=(0,),
                    m_h=1, n_h=0,
                    m_l=0, n_l=0
                )
            ),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(
                rank=0,
                m=0, n=0,
                m_l=0, n_l=0,
                m_t=[0], n_t=[0],
                m_h=0, n_h=0,
                m_r=0, n_r=0
            ),
            (
                et.disconnected_t_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_lhs=0, n_lhs=0,
                    m_l=0, n_l=0,
                    m_h=0, n_h=0,
                    m_r=0, n_r=1
                ),
            ),
            et.connected_eT_h_z_operator_namedtuple(
                rank=0,
                m=0, n=0,
                m_lhs=0, n_lhs=0,
                m_t=[0], n_t=[0],
                m_l=0, n_l=0,
                m_r=0, n_r=0
            ),
            (
                None,
                et.disconnected_eT_z_right_operator_namedtuple(
                    rank=1,
                    m=1, n=0,
                    m_lhs=0, n_lhs=0,
                    m_t=(1,), n_t=(0,),
                    m_h=0, n_h=0,
                    m_l=0, n_l=0
                )
            ),
            1
        ],
        [
            et.connected_eT_lhs_operator_namedtuple(
                rank=0,
                m=0, n=0,
                m_l=0, n_l=0,
                m_t=[0], n_t=[0],
                m_h=0, n_h=0,
                m_r=0, n_r=0
            ),
            (
                et.connected_t_operator_namedtuple(
                    rank=1,
                    m=0, n=1,
                    m_lhs=0, n_lhs=0,
                    m_l=0, n_l=0,
                    m_h=0, n_h=1,
                    m_r=0, n_r=0
                ),
            ),
            et.connected_eT_h_z_operator_namedtuple(
                rank=1,
                m=1, n=0,
                m_lhs=0, n_lhs=0,
                m_t=[1], n_t=[0],
                m_l=0, n_l=0,
                m_r=0, n_r=0
            ),
            (
                None,
                et.disconnected_eT_z_right_operator_namedtuple(
                    rank=0,
                    m=0, n=0,
                    m_lhs=0, n_lhs=0,
                    m_t=(0,), n_t=(0,),
                    m_h=0, n_h=0,
                    m_l=0, n_l=0
                )
            ),
            1
        ]
    ]

    def test_basic(self):
        function_output = et._prepare_third_eTz_latex(
            Test_prepare_third_eTz_latex.term_list,
            split_width=5,
            remove_f_terms=False,
            print_prefactors=True,
            suppress_duplicates=True
        )
        expected_result = str(
            '(\\mathds1\\bh_0\\bz_0 + \\bar{f}\\mathds1\\bh^{}_{\\blue{k}\\red{}}\\bz^{\\blue{k}\\red{}}_{} +' +
            ' \\bar{f}\\bt^{}_{\\magenta{}\\blue{}\\magenta{k}\\red{}}\\bh_0\\bz^{\\magenta{k}\\red{}}_{} + ' +
            '\\bar{f}\\bt^{}_{\\magenta{}\\blue{k}\\magenta{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{}\\bz_0)'
        )
        assert function_output == expected_result

    def test_f_terms_and_dupes(self):  # file flag
        function_output = et._prepare_third_eTz_latex(
            Test_prepare_third_eTz_latex.term_list,
            split_width=5,
            remove_f_terms=True,
            print_prefactors=True,
            suppress_duplicates=False
        )
        expected_result = str(
            '(\\mathds1\\bh_0\\bz_0 + \\bar{f}\\mathds1\\bh^{}_{\\blue{k}\\red{}}\\bz^{\\blue{k}\\red{}}_{} +' +
            ' \\bar{f}\\bt^{}_{\\magenta{}\\blue{}\\magenta{k}\\red{}}\\bh_0\\bz^{\\magenta{k}\\red{}}_{} + \\bar' +
            '{f}\\bt^{}_{\\magenta{}\\blue{k}\\magenta{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{}\\bz_0)'
        )
        assert function_output == expected_result

    def test_long_line_splitting(self):
        term_list = [  # file flag
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
                (zero_disconnected_t_op_nt,),
                zero_con_h_op_nt,
                (None, et.disconnected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
                (zero_disconnected_t_op_nt,),
                et.connected_eT_h_z_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
                (None, et.connected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
                (zero_disconnected_t_op_nt,),
                et.connected_eT_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
                (None, zero_disconnected_z_right),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
                (zero_disconnected_t_op_nt,),
                et.connected_eT_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
                (None, et.connected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_t=(0,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
                (et.disconnected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=1),),
                zero_con_h_op_nt,
                (None, et.disconnected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=(1,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
                (et.disconnected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=1),),
                et.connected_eT_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
                (None, et.disconnected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_t=(1,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
                (et.connected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=1, m_r=0, n_r=0),),
                et.connected_eT_h_z_operator_namedtuple(rank=1, m=1, n=0, m_lhs=0, n_lhs=0, m_t=[1], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
                (None, et.disconnected_eT_z_right_operator_namedtuple(rank=1, m=1, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=0, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
                (et.disconnected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=0, m_r=0, n_r=1),),
                et.connected_eT_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=1, n_lhs=0, m_t=[0], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
                (None, et.connected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=0, n_lhs=0, m_t=(1,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=0, m_r=0, n_r=1),
                (et.connected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=1, m_r=0, n_r=0),),
                et.connected_eT_h_z_operator_namedtuple(rank=2, m=1, n=1, m_lhs=0, n_lhs=0, m_t=[1], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=1),
                (None, et.connected_eT_z_right_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=(0,), n_t=(0,), m_h=1, n_h=0, m_l=0, n_l=0)),
                1
            ],
            [
                et.connected_eT_lhs_operator_namedtuple(rank=1, m=0, n=1, m_l=0, n_l=0, m_t=[0], n_t=[0], m_h=0, n_h=1, m_r=0, n_r=0),
                (et.connected_t_operator_namedtuple(rank=1, m=0, n=1, m_lhs=0, n_lhs=0, m_l=0, n_l=0, m_h=0, n_h=1, m_r=0, n_r=0),),
                et.connected_eT_h_z_operator_namedtuple(rank=2, m=2, n=0, m_lhs=1, n_lhs=0, m_t=[1], n_t=[0], m_l=0, n_l=0, m_r=0, n_r=0),
                (None, zero_disconnected_z_right),
                1
            ]
        ]

        function_output = et._prepare_third_eTz_latex(
            term_list,
            split_width=5,
            remove_f_terms=False,
            print_prefactors=True,
            suppress_duplicates=True
        )
        expected_result = str(
            '\n\\mathds1\\bh_0\\bz^{\\red{y}}_{} + \\bar{f}\\frac{(2)}{2!}\\mathds1\\bh^{}_{\\blue{k}\\red{}}\\bz^{\\b' +
            'lue{k}\\red{y}}_{} + \\mathds1\\bh^{\\blue{}\\red{y}}_{}\\bz_0 + \\bar{f}\\mathds1\\bh^{\\blue{}\\red{y}}_'+
            '{\\blue{k}\\red{}}\\bz^{\\blue{k}\\red{}}_{} + \\bar{f}\\frac{(2)}{2!}\\bt^{}_{\\magenta{}\\blue{}\\magent'+
            'a{k}\\red{}}\\bh_0\\bz^{\\magenta{k}\\red{y}}_{}\n    \\\\  &+  % split long equation\n\\bar{f}\\bt^{}_{\\'+
            'magenta{}\\blue{}\\magenta{k}\\red{}}\\bh^{\\blue{}\\blue{}\\red{y}}_{}\\bz^{\\magenta{k}\\red{}}_{} + \\b'+
            'ar{f}\\bt^{}_{\\magenta{}\\blue{k}\\magenta{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{}\\bz^{\\magenta{}\\'+
            'red{y}}_{} + \\bar{f}^{2}\\frac{(2)}{2!}\\bt^{}_{\\magenta{}\\blue{}\\magenta{k}\\red{}}\\bh^{\\blue{}\\bl'+
            'ue{}\\red{y}}_{\\blue{l}\\red{}}\\bz^{\\magenta{k}\\blue{l}\\red{}}_{} + \\bar{f}^{2}\\frac{(2)}{2!}\\bt^{'+
            '}_{\\magenta{}\\blue{k}\\magenta{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{\\blue{l}\\red{}}\\bz^{\\magent'+
            'a{}\\blue{l}\\red{y}}_{} + \\bar{f}\\frac{(2)}{2!}\\bt^{}_{\\magenta{}\\blue{k}\\magenta{}\\red{}}\\bh^{\\'+
            'blue{k}\\blue{}\\red{y}}_{}\\bz_0\n'
        )
        assert function_output == expected_result


class Test_prepare_eTz_z_terms:

    def test_basic(self):
        Z_left = None
        Z_right = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = et._prepare_eTz_z_terms(Z_left, Z_right, zhz_debug=False)
        expected_result = [
            (
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0)
            ),
            (
                none_gen_op_nt,
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            )
        ]
        assert function_output == expected_result

    # TODO add tests for ZH and ZHZ terms once supported


class Test_prepare_eTz_T_terms:

    def test_basic_op(self):
        eT_series_term = et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)
        function_output = et._prepare_eTz_T_terms(eT_series_term)
        expected_result = ([[et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)]], 'n')
        assert function_output == expected_result

    def test_T_sup_1_op(self):
        eT_series_term = [et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)]
        function_output = et._prepare_eTz_T_terms(eT_series_term)
        expected_result = ([[et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)]], 1)
        assert function_output == expected_result


class Test_filter_out_valid_eTz_terms:

    def test_basic(self):
        LHS = zero_gen_op_nt
        eT = et.general_operator_namedtuple(name='1', rank=0, m=0, n=0)
        H = et.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zero_h_op_nt,
                fcc.h_operator_namedtuple(rank=1, m=0, n=1),
                fcc.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z_left = None
        Z_right = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        total_list = []
        function_output = et._filter_out_valid_eTz_terms(LHS, eT, H, Z_left, Z_right, total_list, zhz_debug=False)
        expected_result = None
        assert function_output == expected_result


class Test_build_third_eTz_term:

    def test_basic(self):
        LHS = et.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        eT_taylor_expansion = [
            et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ]
        ]
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zero_h_op_nt,
                fcc.h_operator_namedtuple(rank=1, m=0, n=1),
                fcc.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = et._build_third_eTz_term(LHS, eT_taylor_expansion, H, Z, remove_f_terms=False)
        expected_result = str(
            '(\\mathds1\\bh_0\\bz^{\\red{y}}_{} + \\mathds1\\bh^{\\blue{}\\red{y}}_{}\\bz_0 + \\bar{f}\\bt^{}_{\\magen'+
            'ta{}\\blue{}\\magenta{k}\\red{}}\\bh^{\\blue{}\\blue{}\\red{y}}_{}\\bz^{\\magenta{k}\\red{}}_{} + \\bar{f'+
            '}\\bt^{}_{\\magenta{}\\blue{k}\\magenta{}\\red{}}\\bh^{\\blue{k}\\blue{}\\red{}}_{}\\bz^{\\magenta{}\\red'+
            '{y}}_{})'
        )
        assert function_output == expected_result


class Test_generate_eT_z_symmetric_latex_equations:

    def test_basic(self):
        LHS = et.general_operator_namedtuple(name='b', rank=1, m=0, n=1)
        eT_taylor_expansion = [
            et.general_operator_namedtuple(name='1', rank=0, m=0, n=0),
            [
                et.general_operator_namedtuple(name='s_1', rank=1, m=0, n=1)
            ]
        ]
        H = nt.hamiltonian_namedtuple(
            maximum_rank=1,
            operator_list=[
                zero_h_op_nt,
                fcc.h_operator_namedtuple(rank=1, m=0, n=1),
                fcc.h_operator_namedtuple(rank=1, m=1, n=0)
            ]
        )
        Z = zhz.z_operator_namedtuple(
            maximum_rank=1,
            operator_list=[
                et.general_operator_namedtuple(name='z', rank=0, m=0, n=0),
                et.general_operator_namedtuple(name='z^1', rank=1, m=1, n=0)
            ]
        )
        function_output = et._generate_eT_z_symmetric_latex_equations(
            LHS,
            eT_taylor_expansion,
            H,
            Z,
            only_ground_state=True,
            remove_f_terms=False
        )
        expected_result = str(
            '\\bh^{i}(1-\\delta_{x\\gamma})\\\\&+\\sum\\Big((\\mathds1\\bh_0\\bz^{\\red{y}} + \\mathds1\\bh^{\\blue{}'+
            '\\red{y}}\\bz_0 + \\bar{f}\\bt_{\\magenta{}\\blue{}\\magenta{k}\\red{}}\\bh^{\\blue{}\\blue{}\\red{y}}\\'+
            'bz^{\\magenta{k}\\red{}} + \\bar{f}\\bt_{\\magenta{}\\blue{k}\\magenta{}\\red{}}\\bh^{\\blue{k}\\blue{}'+
            '\\red{}}\\bz^{\\magenta{}\\red{y}})\\Big)(1-\\delta_{cb})\\\\&-i\\sum\\Big(\\dv{\\hat{\\bt}^{i}_{\\gamma'+
            '}}{\\tau}\\,\\hat{\\bz}_{0, \\gamma} + \\dv{\\hat{\\bt}_{0, \\gamma}}{\\tau}\\,\\hat{\\bz}^{i}_{\\gamma}'+
            '\\Big)'
        )
        assert function_output == expected_result


class Test_run_main_et_zhz:

    def test_run_main(self):
        et.generate_eT_z_t_symmetric_latex(
            [1, 1, 1, 1, 1],
            only_ground_state=True,
            remove_f_terms=False,
            path="./generated_latex.tex"
        )
