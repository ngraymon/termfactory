# system imports
# import re
import pytest

# local imports
from . import context
import code_w_equations as cw


class Test_gen_w_oper:

    def test_generate_w_operator_prefactor_basic(self):
        function_output = cw._generate_w_operator_prefactor((1, 1, 1))
        expected_result = '1/factorial(3)'
        assert function_output == expected_result

    def test_generate_w_operator_prefactor_max_1(self):
        function_output = cw._generate_w_operator_prefactor((1,))
        expected_result = ''
        assert function_output == expected_result

    def test_generate_w_operator_prefactor_single(self):
        """tuple of len 1 but not containing 1"""
        function_output = cw._generate_w_operator_prefactor((3,))
        expected_result = f"1/factorial({3})"
        assert function_output == expected_result

    def test_generate_w_operator_prefactor_single_else(self):
        """else case"""
        function_output = cw._generate_w_operator_prefactor((2, 1))
        expected_result = '1/(factorial(2) * factorial(2))'
        assert function_output == expected_result


class Test_begin_code_gen:

    def test_generate_surface_index(self):
        partition = (1, 1, 1)
        function_output = cw._generate_surface_index(partition)
        expected_result = ['ac', 'cd', 'db']
        assert function_output == expected_result

    def test_generate_mode_index(self):
        partition = (1, 1, 1)
        order = 3
        function_output = cw._generate_mode_index(partition, order)
        expected_result = [['i', 'j', 'k']]
        assert function_output == expected_result

    def test_w_einsum_list(self):
        partition = (1, 1, 1)
        order = 3
        function_output = cw._w_einsum_list(partition, order)
        expected_result = ["np.einsum('aci, cdj, dbk->abijk', t_i, t_i, t_i)"]
        assert function_output == expected_result

    def test_optimized_w_einsum_list(self):
        partition = (1, 1, 1)
        order = 3
        function_output = cw._optimized_w_einsum_list(partition, order, iterator_name='optimized_einsum')
        expected_result = ['next(optimized_einsum)(t_i, t_i, t_i)']
        assert function_output == expected_result


class Test_contributions:

    def test_construct_vemx_contributions_definition(self):
        """basic test, not opt_einsum"""
        return_string = ''
        order = 3
        function_output = cw._construct_vemx_contributions_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum')
        expected_result = '\ndef _add_order_3_vemx_contributions(W_3, t_args, truncation):\n    """Calculate the order 3 VECI/CC (mixed) contributions to the W operator\n    for use in the calculation of the residuals.\n    """\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n'
        assert function_output == expected_result

    def test_construct_vemx_contributions_definition_einsum(self):
        """test opt_einsum"""
        return_string = ''
        order = 3
        function_output = cw._construct_vemx_contributions_definition(return_string, order, opt_einsum=True, iterator_name='optimized_einsum')
        expected_result = '\ndef _add_order_3_vemx_contributions_optimized(W_3, t_args, truncation, opt_path_list):\n    """Calculate the order 3 VECI/CC (mixed) contributions to the W operator\n    for use in the calculation of the residuals.\n    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n    """\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n    # make an iterable out of the `opt_path_list`\n    optimized_einsum = iter(opt_path_list)\n'
        assert function_output == expected_result

    def test_generate_vemx_contributions_basic(self):
        """basic test"""
        order = 3
        function_output = cw._generate_vemx_contributions(order, opt_einsum=False)
        expected_result = '\ndef _add_order_3_vemx_contributions(W_3, t_args, truncation):\n    """Calculate the order 3 VECI/CC (mixed) contributions to the W operator\n    for use in the calculation of the residuals.\n    """\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n    # DOUBLES contribution\n    if truncation.doubles:\n        W_3 += 1/(factorial(2) * factorial(2)) * (\n            np.einsum(\'aci, cbjk->abijk\', t_i, t_ij) +\n            np.einsum(\'acij, cbk->abijk\', t_ij, t_i)\n        )\n    # SINGLES contribution\n    W_3 += 1/factorial(3) * (np.einsum(\'aci, cdj, dbk->abijk\', t_i, t_i, t_i))\n    return\n'
        assert function_output == expected_result

    def test_generate_vemx_contributions_small_order(self):
        """test on small order"""
        order = 1
        function_output = cw._generate_vemx_contributions(order, opt_einsum=False)
        expected_result = '\ndef _add_order_1_vemx_contributions(W_1, t_args, truncation):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n        "which requires a W operator of at least 2nd order"\n    )\n'
        assert function_output == expected_result

    def test_construct_vecc_contributions_definition_basic(self):
        """basic test"""
        return_string = ""
        order = 2
        function_output = cw._construct_vecc_contributions_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum')
        expected_result = '\ndef _add_order_2_vecc_contributions(W_2, t_args, truncation):\n    """Calculate the order 2 VECC contributions to the W operator\n    for use in the calculation of the residuals.\n    """\n    # unpack the `t_args`\n    *unusedargs = t_args\n'
        assert function_output == expected_result

    def test_construct_vecc_contributions_definition_einsum(self):
        """einsum test"""
        return_string = ""
        order = 2
        function_output = cw._construct_vecc_contributions_definition(return_string, order, opt_einsum=True, iterator_name='optimized_einsum')
        expected_result = '\ndef _add_order_2_vecc_contributions_optimized(W_2, t_args, truncation, opt_path_list):\n    """Calculate the order 2 VECC contributions to the W operator\n    "for use in the calculation of the residuals.\n    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n    """\n    # unpack the `t_args`\n    *unusedargs = t_args\n    # make an iterable out of the `opt_path_list`\n    optimized_einsum = iter(opt_path_list)\n'
        assert function_output == expected_result

    def test_generate_vecc_contributions_basic(self):
        """basic test"""
        order = 2
        function_output = cw._generate_vecc_contributions(order, opt_einsum=False)
        expected_result = '\ndef _add_order_2_vecc_contributions(W_2, t_args, truncation):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n        "which requires a W operator of at least 4th order"\n    )\n'
        assert function_output == expected_result

    def test_generate_vecc_contributions_basic_high_order(self):
        """high order test"""
        order = 5
        function_output = cw._generate_vecc_contributions(order, opt_einsum=False)
        expected_result = '\ndef _add_order_5_vecc_contributions(W_5, t_args, truncation):\n    """Calculate the order 5 VECC contributions to the W operator\n    for use in the calculation of the residuals.\n    """\n    # unpack the `t_args`\n    t_i, t_ij, t_ijk, *unusedargs = t_args\n    # TRIPLES contribution\n    if truncation.triples:\n        W_5 += 1/(factorial(2) * factorial(3) * factorial(2)) * (\n            np.einsum(\'acij, cbklm->abijklm\', t_ij, t_ijk) +\n            np.einsum(\'acijk, cblm->abijklm\', t_ijk, t_ij)\n        )\n    # DOUBLES contribution\n    if truncation.doubles:\n        W_5 += 1/(factorial(3) * factorial(2) * factorial(2)) * (\n            np.einsum(\'aci, cdjk, dblm->abijklm\', t_i, t_ij, t_ij) +\n            np.einsum(\'acij, cdk, dblm->abijklm\', t_ij, t_i, t_ij) +\n            np.einsum(\'acij, cdkl, dbm->abijklm\', t_ij, t_ij, t_i)\n        )\n    return\n'
        assert function_output == expected_result


class Test_def_w_func:

    def test_construct_w_function_definition(self):
        """basic test"""
        return_string = ""
        order = 2
        function_output = cw._construct_w_function_definition(return_string, order, opt_einsum=False, iterator_name='optimized_einsum')
        expected_result = '\ndef _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation):\n    """Calculate the order 2 W operator for use in the calculation of the residuals."""\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n'
        assert function_output == expected_result

    def test_construct_w_function_definition_einsum(self):
        """einsum test"""
        return_string = ""
        order = 2
        function_output = cw._construct_w_function_definition(return_string, order, opt_einsum=True, iterator_name='optimized_einsum')
        expected_result = '\ndef _calculate_order_2_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_opt_path_list, vecc_opt_path_list):\n    """Calculate the order 2 W operator for use in the calculation of the residuals.\n    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n    """\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n'
        assert function_output == expected_result

    def test_write_w_function_strings_basic(self):
        """basic test"""
        order = 2
        function_output = cw._write_w_function_strings(order, opt_einsum=False)
        expected_result = '\ndef _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation):\n    """Calculate the order 2 W operator for use in the calculation of the residuals."""\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n    # Creating the 2nd order W operator\n    W_2 = np.zeros((A, A, N, N), dtype=complex)\n\n    # add the VECI contribution\n    if truncation.doubles:\n        W_2 += 1/factorial(2) * t_ij\n    if ansatz.VE_MIXED:\n        _add_order_2_vemx_contributions(W_2, t_args, truncation)\n    elif ansatz.VECC:\n        _add_order_2_vemx_contributions(W_2, t_args, truncation)\n        pass  # no VECC contributions for order < 4\n\n    # Symmetrize the W operator\n    symmetric_w = symmetrize_tensor(N, W_2, order=2)\n    return symmetric_w\n'
        assert function_output == expected_result

    def test_write_w_function_strings_zero_case(self):
        """zero test"""
        order = 0
        with pytest.raises(Exception, match="We should not call `_write_w_function_strings` with order 0."):
            cw._write_w_function_strings(order, opt_einsum=False)

    def test_write_w_function_strings_1st_order(self):
        """1st order test"""
        order = 1
        function_output = cw._write_w_function_strings(order, opt_einsum=False)
        expected_result = '\ndef _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation):\n    """Calculate the order 1 W operator for use in the calculation of the residuals."""\n    # unpack the `t_args`\n    t_i, *unusedargs = t_args\n    # Creating the 1st order W operator\n    W_1 = np.zeros((A, A, N), dtype=complex)\n    # Singles contribution\n    W_1 += t_i\n    return W_1\n'
        assert function_output == expected_result

    def test_write_w_function_strings_high_order(self):
        """high order test"""
        order = 5
        function_output = cw._write_w_function_strings(order, opt_einsum=False)
        expected_result = '\ndef _calculate_order_5_w_operator(A, N, t_args, ansatz, truncation):\n    """Calculate the order 5 W operator for use in the calculation of the residuals."""\n    # unpack the `t_args`\n    t_i, t_ij, t_ijk, t_ijkl, t_ijklm, *unusedargs = t_args\n    # Creating the 5th order W operator\n    W_5 = np.zeros((A, A, N, N, N, N, N), dtype=complex)\n\n    # add the VECI contribution\n    if truncation.quintuples:\n        W_5 += 1/factorial(5) * t_ijklm\n    if ansatz.VE_MIXED:\n        _add_order_5_vemx_contributions(W_5, t_args, truncation)\n    elif ansatz.VECC:\n        _add_order_5_vemx_contributions(W_5, t_args, truncation)\n        _add_order_5_vecc_contributions(W_5, t_args, truncation)\n\n    # Symmetrize the W operator\n    symmetric_w = symmetrize_tensor(N, W_5, order=5)\n    return symmetric_w\n'
        assert function_output == expected_result

    def test_write_master_w_compute_function_basic(self):
        """basic test"""
        max_order = 2
        function_output = cw._write_master_w_compute_function(max_order, opt_einsum=False)
        expected_result = '\ndef compute_w_operators(A, N, t_args, ansatz, truncation):\n    """Compute a number of W operators depending on the level of truncation."""\n\n    if not truncation.singles:\n        raise Exception(\n            "It appears that `singles` is not true, this cannot be.\\n"\n            "Something went terribly wrong!!!\\n\\n"\n            f"{truncation}\\n"\n        )\n\n    w_1 = _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation)\n    w_2 = _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation)\n    w_3 = _calculate_order_3_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.doubles:\n        return w_1, w_2, w_3, None, None, None\n    else:\n        w_4 = _calculate_order_4_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.triples:\n        return w_1, w_2, w_3, w_4, None, None\n    else:\n        w_5 = _calculate_order_5_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.quadruples:\n        return w_1, w_2, w_3, w_4, w_5, None\n    else:\n        w_6 = _calculate_order_6_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.quintuples:\n        return w_1, w_2, w_3, w_4, w_5, w_6\n    else:\n        raise Exception(\n            "Attempting to calculate W^7 operator (quintuples)\\n"\n            "This is currently not implemented!!\\n"\n        )\n'
        assert function_output == expected_result

    def test_write_master_w_compute_function_einsum(self):
        """einsum test"""
        max_order = 2
        function_output = cw._write_master_w_compute_function(max_order, opt_einsum=True)
        expected_result = '\ndef compute_w_operators_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths, vecc_optimized_paths):\n    """Compute a number of W operators depending on the level of truncation."""\n\n    if not truncation.singles:\n        raise Exception(\n            "It appears that `singles` is not true, this cannot be.\\n"\n            "Something went terribly wrong!!!\\n\\n"\n            f"{truncation}\\n"\n        )\n\n    w_1 = _calculate_order_1_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[0], vecc_optimized_paths[0])\n    w_2 = _calculate_order_2_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[1], vecc_optimized_paths[1])\n    w_3 = _calculate_order_3_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[2], vecc_optimized_paths[2])\n\n    if not truncation.doubles:\n        return w_1, w_2, w_3, None, None, None\n    else:\n        w_4 = _calculate_order_4_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[3], vecc_optimized_paths[3])\n\n    if not truncation.triples:\n        return w_1, w_2, w_3, w_4, None, None\n    else:\n        w_5 = _calculate_order_5_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[4], vecc_optimized_paths[4])\n\n    if not truncation.quadruples:\n        return w_1, w_2, w_3, w_4, w_5, None\n    else:\n        w_6 = _calculate_order_6_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[5], vecc_optimized_paths[5])\n\n    if not truncation.quintuples:\n        return w_1, w_2, w_3, w_4, w_5, w_6\n    else:\n        raise Exception(\n            "Attempting to calculate W^7 operator (quintuples)\\n"\n            "This is currently not implemented!!\\n"\n        )\n'
        assert function_output == expected_result


class Test_optimizers:

    def test_t_term_shape_string(self):
        order = 2
        function_output = cw._t_term_shape_string(order)
        expected_result = '(A, A, N, N)'
        assert function_output == expected_result

    def test_contracted_expressions(self):
        partition_list = [(2, 1), (1, 1, 1)]
        order = 3
        function_output = cw._contracted_expressions(partition_list, order)
        expected_result = [[2, "oe.contract_expression('aci, cbjk->abijk', (A, A, N), (A, A, N, N)),\n", "oe.contract_expression('acij, cbk->abijk', (A, A, N, N), (A, A, N)),\n"], [1, "oe.contract_expression('aci, cdj, dbk->abijk', (A, A, N), (A, A, N), (A, A, N)),\n"]]
        assert function_output == expected_result

    def test_write_optimized_vemx_paths_function(self):
        max_order = 2
        function_output = cw._write_optimized_vemx_paths_function(max_order)
        expected_result = '\ndef compute_optimized_vemx_paths(A, N, truncation):\n    """Calculate optimized paths for the VECI/CC (mixed) einsum calls up to `highest_order`."""\n\n    order_2_list, order_3_list = [], []\n    order_4_list, order_5_list, order_6_list = [], [], []\n\n    if truncation.singles:\n        order_2_list.extend([\n            oe.contract_expression(\'aci, cbj->abij\', (A, A, N), (A, A, N)),\n        ])\n\n\n    return [[], order_2_list]\n'
        assert function_output == expected_result

    def test_write_optimized_vecc_paths_function(self):
        max_order = 2
        function_output = cw._write_optimized_vecc_paths_function(max_order)
        expected_result = '\ndef compute_optimized_vecc_paths(A, N, truncation):\n    """Calculate optimized paths for the VECC einsum calls up to `highest_order`."""\n\n    order_4_list, order_5_list, order_6_list = [], [], []\n\n    if not truncation.doubles:\n        log.warning(\'Did not calculate optimized VECC paths of the dt amplitudes\')\n        return [[], [], [], [], [], []]\n\n\n    return [[], [], []]\n'
        assert function_output == expected_result

    def test_write_optimized_vecc_paths_function_high_order(self):
        max_order = 5
        function_output = cw._write_optimized_vecc_paths_function(max_order)
        expected_result = '\ndef compute_optimized_vecc_paths(A, N, truncation):\n    """Calculate optimized paths for the VECC einsum calls up to `highest_order`."""\n\n    order_4_list, order_5_list, order_6_list = [], [], []\n\n    if not truncation.doubles:\n        log.warning(\'Did not calculate optimized VECC paths of the dt amplitudes\')\n        return [[], [], [], [], [], []]\n\n    if truncation.doubles:\n        order_4_list.extend([\n            oe.contract_expression(\'acij, cbkl->abijkl\', (A, A, N, N), (A, A, N, N)),\n        ])\n\n    if truncation.triples:\n        order_5_list.extend([\n            oe.contract_expression(\'acij, cbklm->abijklm\', (A, A, N, N), (A, A, N, N, N)),\n            oe.contract_expression(\'acijk, cblm->abijklm\', (A, A, N, N, N), (A, A, N, N)),\n        ])\n\n    if truncation.doubles:\n        order_5_list.extend([\n            oe.contract_expression(\'aci, cdjk, dblm->abijklm\', (A, A, N), (A, A, N, N), (A, A, N, N)),\n            oe.contract_expression(\'acij, cdk, dblm->abijklm\', (A, A, N, N), (A, A, N), (A, A, N, N)),\n            oe.contract_expression(\'acij, cdkl, dbm->abijklm\', (A, A, N, N), (A, A, N, N), (A, A, N)),\n        ])\n\n\n    return [[], [], [], order_4_list, order_5_list]\n'
        assert function_output == expected_result


class Test_main_w_eqn_code:

    def test_generate_w_operators_string(self):
        max_order = 2
        function_output = cw.generate_w_operators_string(max_order, s1=75, s2=28)
        expected_result = '# --------------------------------------------------------------------------- #\n# ---------------------------- DEFAULT FUNCTIONS ---------------------------- #\n# --------------------------------------------------------------------------- #\n\n# ---------------------------- VECI/CC CONTRIBUTIONS ---------------------------- #\n\ndef _add_order_1_vemx_contributions(W_1, t_args, truncation):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n        "which requires a W operator of at least 2nd order"\n    )\n\ndef _add_order_2_vemx_contributions(W_2, t_args, truncation):\n    """Calculate the order 2 VECI/CC (mixed) contributions to the W operator\n    for use in the calculation of the residuals.\n    """\n    # unpack the `t_args`\n    t_i, *unusedargs = t_args\n    # SINGLES contribution\n    W_2 += 1/factorial(2) * (np.einsum(\'aci, cbj->abij\', t_i, t_i))\n    return\n\n# ---------------------------- VECC CONTRIBUTIONS ---------------------------- #\n\ndef _add_order_1_vecc_contributions(W_1, t_args, truncation):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n        "which requires a W operator of at least 4th order"\n    )\n\ndef _add_order_2_vecc_contributions(W_2, t_args, truncation):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n        "which requires a W operator of at least 4th order"\n    )\n\n# ---------------------------- W OPERATOR FUNCTIONS ---------------------------- #\n\ndef _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation):\n    """Calculate the order 1 W operator for use in the calculation of the residuals."""\n    # unpack the `t_args`\n    t_i, *unusedargs = t_args\n    # Creating the 1st order W operator\n    W_1 = np.zeros((A, A, N), dtype=complex)\n    # Singles contribution\n    W_1 += t_i\n    return W_1\n\ndef _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation):\n    """Calculate the order 2 W operator for use in the calculation of the residuals."""\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n    # Creating the 2nd order W operator\n    W_2 = np.zeros((A, A, N, N), dtype=complex)\n\n    # add the VECI contribution\n    if truncation.doubles:\n        W_2 += 1/factorial(2) * t_ij\n    if ansatz.VE_MIXED:\n        _add_order_2_vemx_contributions(W_2, t_args, truncation)\n    elif ansatz.VECC:\n        _add_order_2_vemx_contributions(W_2, t_args, truncation)\n        pass  # no VECC contributions for order < 4\n\n    # Symmetrize the W operator\n    symmetric_w = symmetrize_tensor(N, W_2, order=2)\n    return symmetric_w\n\ndef compute_w_operators(A, N, t_args, ansatz, truncation):\n    """Compute a number of W operators depending on the level of truncation."""\n\n    if not truncation.singles:\n        raise Exception(\n            "It appears that `singles` is not true, this cannot be.\\n"\n            "Something went terribly wrong!!!\\n\\n"\n            f"{truncation}\\n"\n        )\n\n    w_1 = _calculate_order_1_w_operator(A, N, t_args, ansatz, truncation)\n    w_2 = _calculate_order_2_w_operator(A, N, t_args, ansatz, truncation)\n    w_3 = _calculate_order_3_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.doubles:\n        return w_1, w_2, w_3, None, None, None\n    else:\n        w_4 = _calculate_order_4_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.triples:\n        return w_1, w_2, w_3, w_4, None, None\n    else:\n        w_5 = _calculate_order_5_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.quadruples:\n        return w_1, w_2, w_3, w_4, w_5, None\n    else:\n        w_6 = _calculate_order_6_w_operator(A, N, t_args, ansatz, truncation)\n\n    if not truncation.quintuples:\n        return w_1, w_2, w_3, w_4, w_5, w_6\n    else:\n        raise Exception(\n            "Attempting to calculate W^7 operator (quintuples)\\n"\n            "This is currently not implemented!!\\n"\n        )\n\n# --------------------------------------------------------------------------- #\n# --------------------------- OPTIMIZED FUNCTIONS --------------------------- #\n# --------------------------------------------------------------------------- #\n\n# ---------------------------- VECI/CC CONTRIBUTIONS ---------------------------- #\n\ndef _add_order_1_vemx_contributions_optimized(W_1, t_args, truncation, opt_path_list):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n        "which requires a W operator of at least 2nd order"\n    )\n\ndef _add_order_2_vemx_contributions_optimized(W_2, t_args, truncation, opt_path_list):\n    """Calculate the order 2 VECI/CC (mixed) contributions to the W operator\n    for use in the calculation of the residuals.\n    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n    """\n    # unpack the `t_args`\n    t_i, *unusedargs = t_args\n    # make an iterable out of the `opt_path_list`\n    optimized_einsum = iter(opt_path_list)\n    # SINGLES contribution\n    W_2 += 1/factorial(2) * (next(optimized_einsum)(t_i, t_i))\n    return\n\n# ---------------------------- VECC CONTRIBUTIONS ---------------------------- #\n\ndef _add_order_1_vecc_contributions_optimized(W_1, t_args, truncation, opt_path_list):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n        "which requires a W operator of at least 4th order"\n    )\n\ndef _add_order_2_vecc_contributions_optimized(W_2, t_args, truncation, opt_path_list):\n    """Exists for error checking."""\n    raise Exception(\n        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n        "which requires a W operator of at least 4th order"\n    )\n\n# ---------------------------- W OPERATOR FUNCTIONS ---------------------------- #\n\ndef _calculate_order_1_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_opt_path_list, vecc_opt_path_list):\n    """Calculate the order 1 W operator for use in the calculation of the residuals.\n    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n    """\n    # unpack the `t_args`\n    t_i, *unusedargs = t_args\n    # Creating the 1st order W operator\n    W_1 = np.zeros((A, A, N), dtype=complex)\n    # Singles contribution\n    W_1 += t_i\n    return W_1\n\ndef _calculate_order_2_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_opt_path_list, vecc_opt_path_list):\n    """Calculate the order 2 W operator for use in the calculation of the residuals.\n    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n    """\n    # unpack the `t_args`\n    t_i, t_ij, *unusedargs = t_args\n    # Creating the 2nd order W operator\n    W_2 = np.zeros((A, A, N, N), dtype=complex)\n\n    # add the VECI contribution\n    if truncation.doubles:\n        W_2 += 1/factorial(2) * t_ij\n    if ansatz.VE_MIXED:\n        _add_order_2_vemx_contributions_optimized(W_2, t_args, truncation, vemx_opt_path_list)\n    elif ansatz.VECC:\n        _add_order_2_vemx_contributions_optimized(W_2, t_args, truncation, vemx_opt_path_list)\n        pass  # no VECC contributions for order < 4\n\n    # Symmetrize the W operator\n    symmetric_w = symmetrize_tensor(N, W_2, order=2)\n    return symmetric_w\n\ndef compute_w_operators_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths, vecc_optimized_paths):\n    """Compute a number of W operators depending on the level of truncation."""\n\n    if not truncation.singles:\n        raise Exception(\n            "It appears that `singles` is not true, this cannot be.\\n"\n            "Something went terribly wrong!!!\\n\\n"\n            f"{truncation}\\n"\n        )\n\n    w_1 = _calculate_order_1_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[0], vecc_optimized_paths[0])\n    w_2 = _calculate_order_2_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[1], vecc_optimized_paths[1])\n    w_3 = _calculate_order_3_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[2], vecc_optimized_paths[2])\n\n    if not truncation.doubles:\n        return w_1, w_2, w_3, None, None, None\n    else:\n        w_4 = _calculate_order_4_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[3], vecc_optimized_paths[3])\n\n    if not truncation.triples:\n        return w_1, w_2, w_3, w_4, None, None\n    else:\n        w_5 = _calculate_order_5_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[4], vecc_optimized_paths[4])\n\n    if not truncation.quadruples:\n        return w_1, w_2, w_3, w_4, w_5, None\n    else:\n        w_6 = _calculate_order_6_w_operator_optimized(A, N, t_args, ansatz, truncation, vemx_optimized_paths[5], vecc_optimized_paths[5])\n\n    if not truncation.quintuples:\n        return w_1, w_2, w_3, w_4, w_5, w_6\n    else:\n        raise Exception(\n            "Attempting to calculate W^7 operator (quintuples)\\n"\n            "This is currently not implemented!!\\n"\n        )\n\n\n# ---------------------------- OPTIMIZED PATHS FUNCTION ---------------------------- #\n\ndef compute_optimized_vemx_paths(A, N, truncation):\n    """Calculate optimized paths for the VECI/CC (mixed) einsum calls up to `highest_order`."""\n\n    order_2_list, order_3_list = [], []\n    order_4_list, order_5_list, order_6_list = [], [], []\n\n    if truncation.singles:\n        order_2_list.extend([\n            oe.contract_expression(\'aci, cbj->abij\', (A, A, N), (A, A, N)),\n        ])\n\n\n    return [[], order_2_list]\n\n\ndef compute_optimized_vecc_paths(A, N, truncation):\n    """Calculate optimized paths for the VECC einsum calls up to `highest_order`."""\n\n    order_4_list, order_5_list, order_6_list = [], [], []\n\n    if not truncation.doubles:\n        log.warning(\'Did not calculate optimized VECC paths of the dt amplitudes\')\n        return [[], [], [], [], [], []]\n\n\n    return [[], [], []]\n\n'
        assert function_output == expected_result

    def test_run_main_w_eqn_func(self):
        # TODO file compare assert
        cw.generate_w_operator_equations_file(2, path="./w_operator_equations.py")
