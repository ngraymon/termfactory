"""A basic test of the multiple modules"""

# system imports
# import os
# from os.path import dirname, join, abspath

# local imports
from .context import latex_zhz as zhz
from .context import latex_full_cc as fcc

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

    @pytest.fixture(scope="class", params=[0, 1, 2])
    def A(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[0, 1, 2])
    def B(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[0, 1, 2])
    def C(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[0, 1, 2])
    def D(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def truncations(self, A, B, C, D):
        return [A, B, C, D]

    def test_ground_state(self, truncations):
        fcc.generate_full_cc_latex(truncations, only_ground_state=True, path="./ground_state_full_cc_equations.tex")

    def test_excited_state(self, truncations):
        fcc.generate_full_cc_latex(truncations, only_ground_state=False, path="./full_cc_equations.tex")


class TestExcitedState():

    @pytest.fixture(params=[0, 1, 2, 3])
    def maximum_h_rank(self, request):
        return request.param

    @pytest.fixture(params=[0, 1, 2])
    def omega_max_order(self, request):
        return request.param

    @pytest.fixture(params=[0, 1, 2])
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
