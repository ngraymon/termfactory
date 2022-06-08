"""A blanket test of the multiple modules"""

# system imports
import re

# local imports
from .context import latex_zhz as zhz
from .context import latex_full_cc as fcc
from .context import latex_eT_zhz as eTzhz
from .context import latex_w_equations as weqn

from .context import code_full_cc as code_fcc
from .context import code_residual_equations as code_res
from .context import code_w_equations as code_weqn
from .context import code_dt_equations as code_dt_eqn

from .context import namedtuple_defines as nt

# third party imports
import pytest


# @pytest.fixture(scope="class", params=[(0, 0), (0, 1)])

class TestClassName():

    # this would be if you wanted to store some files to check against
    # test_path = join(abspath(dirname(__file__)), "test_models/")

    def test_func_name(self):
        return

    @pytest.fixture(params=[0, 1, 2])
    def ground_Z(request):
        return zhz.generate_z_operator(maximum_cc_rank=request, only_ground_state=True)


# for testing of 2nd and 4th terms of zhz excited:
def blank_function():
    with pytest.raises('<YourException>') as exc_info:
        '<your code that should raise YourException>'
        pass

    exception_raised = exc_info.value
    '<do asserts here>'


class TestFullccLatex():

    # should add remove_f_terms arg to the file `latex_full_cc.py` so that you can pass it in here

    # add a bunch of fixtures for the truncations and maybe they are similar to the fixtures in `ExcitedStateTesting`
    # and then you can move them to a global scope, you'll have to test this

    def test_zero_value_ground_state(self):
        # check that it fails with the specific assert
        # `Truncations need to be positive integers`
        with pytest.raises(AssertionError,  match="Truncations need to be positive integers") as exc_info:
            fcc.generate_full_cc_latex((0, 1, 1, 1), only_ground_state=True, path="./ground_state_full_cc_equations.tex")
            # exception_raised = exc_info.value

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def A(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def B(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def C(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def D(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def truncations(self, A, B, C, D):
        return [A, B, C, D]

    def test_ground_state(self, truncations):
        fcc.generate_full_cc_latex(truncations, only_ground_state=True, path="./ground_state_full_cc_equations.tex")

    def test_excited_state(self, truncations):
        fcc.generate_full_cc_latex(truncations, only_ground_state=False, path="./full_cc_equations.tex")

    def test_m_h_omega_cannot_equal_h_n_o(self): #ie h_kwargs['n_o'] != o_kwargs['m_h']
        #ADD RAISE EXCEPTION, AS NON-BLANKET VERSION WILL BREAK THIS LATER
        fcc._generate_explicit_connections(nt.general_operator_namedtuple('b', 1, 1, 1), fcc.h_operator_namedtuple(2, 2, 2), {(fcc.connected_namedtuple(0, 1, 0, 0),)})

    def test_sep_s_terms_by_connection_linear_exception(self):
        with pytest.raises(Exception,  match="Linear terms should always be connected or disconnected") as exc_info:
            #total_list=[[fcc.connected_omega_operator_namedtuple(1, 0, 0, 0, 0, [0], [0]),fcc.connected_h_operator_namedtuple(0, 0, 0, 0, 0, [0], [0]),fcc.disconnected_namedtuple((0, 0, 0, 0))],[],[]]
            total_list=[[fcc.connected_omega_operator_namedtuple(1, 1, 1, 1, 1, [1], [1]),
                        fcc.connected_h_operator_namedtuple(1, 0, 0, 0, 0, [0], [0]),
                        (fcc.disconnected_namedtuple(1, 1, 2, 1),)],
                        [fcc.connected_omega_operator_namedtuple(1, 0, 0, 0, 0, [0], [0]),
                        fcc.connected_h_operator_namedtuple(1, 0, 1, 0, 0, [0], [1]),
                        (fcc.connected_namedtuple(1, 0, 0, 0),)],
                        [fcc.connected_omega_operator_namedtuple(0, 0, 0, 0, 0, [0], [0]),
                        fcc.connected_h_operator_namedtuple(1, 1, 0, 0, 0, [1], [0]),
                        (fcc.connected_namedtuple(0, 1, 0, 0),)]]
            fcc._seperate_s_terms_by_connection(total_list)

    def test_sep_s_terms_by_connection_tuple_except(self):
        with pytest.raises(Exception,  match=re.escape("term contains something other than connected/disconnected namedtuple??\n")) as exc_info:
                total_list=[[(1, 1, 1, 1, 1, [1], [1]),
                (1, 0, 0, 0, 0, [0], [0]),
                ((1,1, 2, 1),)],
                [(1, 0, 0, 0, 0, [0], [0]),
                (1, 0, 1, 0, 0, [0], [1]),
                ((1, 0, 0, 0),)],
                [(0, 0, 0, 0, 0, [0], [0]),
                (1, 1, 0, 0, 0, [1], [0]),
                ((0, 1, 0, 0),)]]
                fcc._seperate_s_terms_by_connection(total_list)

    def test_simplify_full_cc_python_prefactor_numerator_greater_than_denom(self):
        fcc._simplify_full_cc_python_prefactor(['2!', '2!', '2!'], ['2!', '2!'])

    def test_simplify_full_cc_python_prefactor_long_list_printer(self):
        fcc._simplify_full_cc_python_prefactor(['2!', '2!', '2!','2!', '2!', '2!'], ['2!', '2!'])

class Test_latex_eT_z_t_ansatz():
    
    @pytest.fixture(scope="class", params=[1, 2])
    def A(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2])
    def B(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2])
    def C(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2])
    def D(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2])
    def E(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def truncations(self, A, B, C, D, E):
        return [A, B, C, D, E]

    def test_ground_state(self, truncations):
        eTzhz.generate_eT_z_t_symmetric_latex(truncations, only_ground_state=True, path="./generated_latex.tex")

    # def test_excited_state(self, truncations):
    #     eTzhz.generate_eT_z_t_symmetric_latex(truncations, only_ground_state=False, path="./generated_latex.tex")

    # add a pytest.raise for Exception: The excited state second eTZH terms are not implemented.

class Test_latex_w_equations:
    @pytest.fixture(scope="class", params=[1, 2, 3, 4])
    def max_w_order(self, request):
        return request.param

    def test_ground_state(self, max_w_order):
        weqn.ground_state_w_equations_latex(max_w_order, path="./ground_state_w_equations.tex")

    def test_excited_state(self, max_w_order):
         weqn.excited_state_w_equations_latex(max_w_order, path="./thermal_w_equations.tex")

    def test_zero_case(self):
        weqn.generate_t_terms_group(nt.w_namedtuple_latex(0,0))

class Test_latex_zhz():

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def A(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def B(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def C(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def D(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def truncations(self, A, B, C, D):
        return [A, B, C, D]

    def test_ground_state(self, truncations):
        zhz.generate_z_t_symmetric_latex(truncations, only_ground_state=True, remove_f_terms=False, path="./generated_latex.tex")

    def test_excited_state(self, truncations):
        not_implemented_yet_message = (
        "The logic for the supporting functions (such as `_filter_out_valid_z_terms` and others)\n"
        "Has only been verified to work for the LHS * H * Z (`third_z`) case.\n"
        "The code may produce some output without halting, but the output is meaningless from a theory standpoint.\n"
        "Do not remove this Exception without consulting with someone else and implementing the requisite functions.")
        with pytest.raises(Exception,  match=re.escape(not_implemented_yet_message)):
            zhz.generate_z_t_symmetric_latex(truncations, only_ground_state=False, remove_f_terms=False, path="./generated_latex.tex")

    # need to add more tests for niche cases
    
class Test_code_fcc():

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def A(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def B(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def C(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def D(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def truncations(self, A, B, C, D):
        return [A, B, C, D]

    def test_fcc_code(self, truncations):
        code_fcc.generate_full_cc_python(truncations, only_ground_state=False, path="./full_cc_equations.py")
    
class Test_code_residuals():

    @pytest.fixture(scope="class", params=[4])
    def max_residual_order(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[2])
    def maximum_h_rank(self, request):
        return request.param
    
    def test_residuals_code(self, max_residual_order,maximum_h_rank):
        code_res.generate_residual_equations_file(max_residual_order, maximum_h_rank, path="./residual_equations.py")

    ## need to resolve key error for certain combinations

class Test_code_w_equations():

    @pytest.fixture(scope="class", params=[1, 2, 3, 5])
    def max_w_order(self, request):
        return request.param

    def test_code_w_equations(self, max_w_order):
        code_weqn.generate_w_operator_equations_file(max_w_order, path="./w_operator_equations.py")

class Test_code_dt_equations():

    @pytest.fixture(scope="class", params=[1, 2, 5])
    def max_w_order(self, request):
        return request.param

    def test_code_dt_equations(self, max_w_order):
        code_dt_eqn.generate_dt_amplitude_equations_file(max_w_order, path="./dt_amplitude_equations.py")
    
class TestExcitedState():

    @pytest.fixture(params=[1, 2, 3])
    def maximum_h_rank(self, request):
        return request.param

    @pytest.fixture(params=[1, 2])
    def omega_max_order(self, request):
        return request.param

    @pytest.fixture(params=[1, 2])
    def maximum_cc_rank(self, request):
        return request.param

    @pytest.fixture()
    def excited_Z(self, maximum_cc_rank):
        return zhz.generate_z_operator(maximum_cc_rank, only_ground_state=False)

    @pytest.fixture()
    def excited_H(self, maximum_h_rank):
        raw_H = zhz.generate_full_cc_hamiltonian_operator(maximum_h_rank)
        pruned_list = [term for term in raw_H.operator_list if (term.rank < 3) or (term.m == 0)]
        H = zhz.hamiltonian_namedtuple(raw_H.maximum_rank, pruned_list)
        return H

    @pytest.fixture()
    def excited_omega(self, maximum_cc_rank, omega_max_order):
        return zhz.generate_omega_operator(maximum_cc_rank, omega_max_order)

    def test_build_second_z_term(self, excited_omega, excited_H, excited_Z):
        for i, omega_term in enumerate(excited_omega.operator_list):
            zhz._build_second_z_term(omega_term, excited_H, excited_Z, remove_f_terms=True)
            zhz._build_second_z_term(omega_term, excited_H, excited_Z, remove_f_terms=False)
