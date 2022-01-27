# system imports
# import pytest

# local imports
from . import context
import code_dt_equations as cdt


class Test_generate_disconnected_einsum_operands_list:

    def test_basic(self):
        dt_index = 0
        tupl = (1, 1)
        function_output = cdt._generate_disconnected_einsum_operands_list(dt_index, tupl)
        expected_result = 'dt_i, t_i'
        assert function_output == expected_result


class Test_generate_disconnected_einsum_function_call_list:

    def test_basic(self):
        partition = (1, 1)
        order = 2
        function_output = cdt._generate_disconnected_einsum_function_call_list(partition, order)
        expected_result = ["np.einsum('aci, cbj->abij', dt_i, t_i)", "np.einsum('aci, cbj->abij', t_i, dt_i)"]
        assert function_output == expected_result


class Test_generate_optimized_disconnected_einsum_function_call_list:

    def test_basic(self):
        partition = (1, 1)
        order = 2
        function_output = cdt._generate_optimized_disconnected_einsum_function_call_list(
            partition,
            order,
            iterator_name='optimized_einsum'
        )
        expected_result = ['next(optimized_einsum)(dt_i, t_i)', 'next(optimized_einsum)(t_i, dt_i)']
        assert function_output == expected_result


class Test_construct_linked_disconnected_definition:

    def test_basic(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'+
            'def _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'+
            '    But not terms (5), (3, 2), (2, 2, 1)\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'+
            'def _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1)\n'+
            '    But not terms (5), (3, 2), (2, 2, 1), (1, 1, 1, 1, 1)\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'
        )
        assert function_output == expected_result


class Test_write_linked_disconnected_strings:

    def test_basic(self):
        order = 2
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'+
            'def _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'+
            '    But not terms (5), (3, 2), (2, 2, 1)\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'+
            '    # Creating the 2nd order return array\n'+
            '    linked_disconnected_terms = np.zeros((A, A, N, N), dtype=complex)\n'+
            '    # the (1, 1) term\n    linked_disconnected_terms += 1/factorial(2) * (\n'+
            '        np.einsum(\'aci, cbj->abij\', dt_i, t_i) +\n'+
            '        np.einsum(\'aci, cbj->abij\', t_i, dt_i)\n'+
            '    )\n'+
            '\n'+
            '    return linked_disconnected_terms\n'
        )
        assert function_output == expected_result

    def test_low_order(self):
        order = 1
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'+
            'def _order_1_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n'+
            '        "which requires a residual of at least 2nd order"\n'+
            '    )\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        order = 2
        function_output = cdt._write_linked_disconnected_strings(order, opt_einsum=True)
        expected_result = str(
            '\n'+
            'def _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1)\n'+
            '    But not terms (5), (3, 2), (2, 2, 1), (1, 1, 1, 1, 1)\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'+
            '    # Creating the 2nd order return array\n'+
            '    linked_disconnected_terms = np.zeros((A, A, N, N), dtype=complex)\n'+
            '    # the (1, 1) term\n    linked_disconnected_terms += 1/factorial(2) * (\n'+
            '        next(optimized_einsum)(dt_i, t_i) +\n'+
            '        next(optimized_einsum)(t_i, dt_i)\n'+
            '    )\n'+
            '\n'+
            '    return linked_disconnected_terms\n'
        )
        assert function_output == expected_result


class Test_construct_un_linked_disconnected_definition:

    def test_basic(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_un_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'+
            'def _order_2_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'+
            '    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_un_linked_disconnected_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'+
            'def _order_2_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'+
            '    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'
        )
        assert function_output == expected_result


class Test_write_un_linked_disconnected_strings:

    def test_high_order(self):
        order = 5
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'+
            'def _order_5_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'+
            '    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, t_ij, t_ijk, t_ijkl, *unusedargs = t_args\n'+
            '    dt_i, dt_ij, dt_ijk, dt_ijkl, *unusedargs = dt_args\n'+
            '    # Creating the 5th order return array\n'+
            '    un_linked_disconnected_terms = np.zeros((A, A, N, N, N, N, N), dtype=complex)\n'+
            '    # the (3, 2) term\n'+
            '    un_linked_disconnected_terms += 1/(factorial(2) * factorial(3) * factorial(2)) * (\n'+
            '        np.einsum(\'acij, cbklm->abijklm\', dt_ij, t_ijk) +\n'+
            '        np.einsum(\'acij, cbklm->abijklm\', t_ij, dt_ijk) +\n'+
            '        np.einsum(\'acijk, cblm->abijklm\', dt_ijk, t_ij) +\n'+
            '        np.einsum(\'acijk, cblm->abijklm\', t_ijk, dt_ij)\n'+
            '    )\n'+
            '    # the (2, 2, 1) term\n'+
            '    un_linked_disconnected_terms += 1/(factorial(3) * factorial(2) * factorial(2)) * (\n'+
            '        np.einsum(\'aci, cdjk, dblm->abijklm\', dt_i, t_ij, t_ij) +\n'+
            '        np.einsum(\'aci, cdjk, dblm->abijklm\', t_i, dt_ij, t_ij) +\n'+
            '        np.einsum(\'aci, cdjk, dblm->abijklm\', t_i, t_ij, dt_ij) +\n'+
            '        np.einsum(\'acij, cdk, dblm->abijklm\', dt_ij, t_i, t_ij) +\n'+
            '        np.einsum(\'acij, cdk, dblm->abijklm\', t_ij, dt_i, t_ij) +\n'+
            '        np.einsum(\'acij, cdk, dblm->abijklm\', t_ij, t_i, dt_ij) +\n'+
            '        np.einsum(\'acij, cdkl, dbm->abijklm\', dt_ij, t_ij, t_i) +\n'+
            '        np.einsum(\'acij, cdkl, dbm->abijklm\', t_ij, dt_ij, t_i) +\n'+
            '        np.einsum(\'acij, cdkl, dbm->abijklm\', t_ij, t_ij, dt_i)\n'+
            '    )\n'+
            '\n'+
            '    return un_linked_disconnected_terms\n'
        )
        assert function_output == expected_result

    def test_high_order_einsum(self):
        order = 5
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=True)
        expected_result = str(
            '\n'+
            'def _order_5_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Calculate all uniquely un-linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (3, 2), (2, 2, 1)\n'+
            '    But not terms (5), (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, t_ij, t_ijk, t_ijkl, *unusedargs = t_args\n'+
            '    dt_i, dt_ij, dt_ijk, dt_ijkl, *unusedargs = dt_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'+
            '    # Creating the 5th order return array\n'+
            '    un_linked_disconnected_terms = np.zeros((A, A, N, N, N, N, N), dtype=complex)\n'+
            '    # the (3, 2) term\n'+
            '    un_linked_disconnected_terms += 1/(factorial(2) * factorial(3) * factorial(2)) * (\n'+
            '        next(optimized_einsum)(dt_ij, t_ijk) +\n'+
            '        next(optimized_einsum)(t_ij, dt_ijk) +\n'+
            '        next(optimized_einsum)(dt_ijk, t_ij) +\n'+
            '        next(optimized_einsum)(t_ijk, dt_ij)\n'+
            '    )\n'+
            '    # the (2, 2, 1) term\n'+
            '    un_linked_disconnected_terms += 1/(factorial(3) * factorial(2) * factorial(2)) * (\n'+
            '        next(optimized_einsum)(dt_i, t_ij, t_ij) +\n'+
            '        next(optimized_einsum)(t_i, dt_ij, t_ij) +\n'+
            '        next(optimized_einsum)(t_i, t_ij, dt_ij) +\n'+
            '        next(optimized_einsum)(dt_ij, t_i, t_ij) +\n'+
            '        next(optimized_einsum)(t_ij, dt_i, t_ij) +\n'+
            '        next(optimized_einsum)(t_ij, t_i, dt_ij) +\n'+
            '        next(optimized_einsum)(dt_ij, t_ij, t_i) +\n'+
            '        next(optimized_einsum)(t_ij, dt_ij, t_i) +\n'+
            '        next(optimized_einsum)(t_ij, t_ij, dt_i)\n'+
            '    )\n'+
            '\n'+
            '    return un_linked_disconnected_terms\n')
        assert function_output == expected_result

    def test_low_order(self):
        order = 2
        function_output = cdt._write_un_linked_disconnected_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'+
            'def _order_2_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'+
            '        "which requires a residual of at least 4th order"\n'+
            '    )\n'
        )
        assert function_output == expected_result


class Test_construct_dt_amplitude_definition:

    def test_basic(self):
        return_string = ''
        order = 2
        function_output = cdt._construct_dt_amplitude_definition(
            return_string,
            order,
            opt_einsum=False,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'+
            'def _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals."""\n'+
            '    # unpack the `w_args`\n'+
            '    w_i, w_ij, *unusedargs = w_args\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        return_string = ''
        order = 3
        function_output = cdt._construct_dt_amplitude_definition(
            return_string,
            order,
            opt_einsum=True,
            iterator_name='optimized_einsum'
        )
        expected_result = str(
            '\n'+
            'def _calculate_order_3_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):\n'+
            '    """Calculate the derivative of the 3 t-amplitude for use in the calculation of the residuals.\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n    # unpack the `w_args`\n    w_i, w_ij, w_ijk, *unusedargs = w_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'
        )
        assert function_output == expected_result


class Test_write_dt_amplitude_strings:

    def test_basic(self):
        order = 2
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'+
            'def _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals."""\n'+
            '    # unpack the `w_args`\n'+
            '    w_i, w_ij, *unusedargs = w_args\n'+
            '    # Calculate the 2nd order residual\n'+
            '    residual = residual_equations.calculate_order_2_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(2) * np.einsum(\'acij,cb->abij\', w_ij, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        residual -= _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n'+
            '    elif ansatz.VECC:\n'+
            '        residual -= _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n'+
            '        pass  # no un-linked disconnected terms for order < 4\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_ij = symmetrize_tensor(N, residual, order=2)\n'+
            '    return dt_ij\n'
        )
        assert function_output == expected_result

    def test_high_order(self):
        order = 5
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=False)
        expected_result = str(
            '\n'+
            'def _calculate_order_5_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Calculate the derivative of the 5 t-amplitude for use in the calculation of the residuals."""\n'+
            '    # unpack the `w_args`\n'+
            '    w_i, w_ij, w_ijk, w_ijkl, w_ijklm, *unusedargs = w_args\n'+
            '    # Calculate the 5th order residual\n'+
            '    residual = residual_equations.calculate_order_5_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(5) * np.einsum(\'acijklm,cb->abijklm\', w_ijklm, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        residual -= _order_5_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n'+
            '    elif ansatz.VECC:\n'+
            '        residual -= _order_5_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n'+
            '        residual -= _order_5_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_ijklm = symmetrize_tensor(N, residual, order=5)\n'+
            '    return dt_ijklm\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        order = 2
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=True)
        expected_result = str(
            '\n'+
            'def _calculate_order_2_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):\n'+
            '    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals.\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `w_args`\n'+
            '    w_i, w_ij, *unusedargs = w_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'+
            '    # Calculate the 2nd order residual\n'+
            '    residual = residual_equations.calculate_order_2_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(2) * opt_epsilon(w_ij, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        residual -= _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n'+
            '    elif ansatz.VECC:\n'+
            '        residual -= _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n'+
            '        pass  # no un-linked disconnected terms for order < 4\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_ij = symmetrize_tensor(N, residual, order=2)\n'+
            '    return dt_ij\n'
        )
        assert function_output == expected_result

    def test_einsum_high_order(self):
        order = 5
        function_output = cdt._write_dt_amplitude_strings(order, opt_einsum=True)
        expected_result = str(
            '\n'+
            'def _calculate_order_5_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):\n'+
            '    """Calculate the derivative of the 5 t-amplitude for use in the calculation of the residuals.\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `w_args`\n'+
            '    w_i, w_ij, w_ijk, w_ijkl, w_ijklm, *unusedargs = w_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'+
            '    # Calculate the 5th order residual\n'+
            '    residual = residual_equations.calculate_order_5_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(5) * opt_epsilon(w_ijklm, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        residual -= _order_5_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n'+
            '    elif ansatz.VECC:\n'+
            '        residual -= _order_5_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n'+
            '        residual -= _order_5_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_ijklm = symmetrize_tensor(N, residual, order=5)\n'+
            '    return dt_ijklm\n'
        )
        assert function_output == expected_result


class Test_write_master_dt_amplitude_function:

    def test_basic(self):
        order = 2
        function_output = cdt._write_master_dt_amplitude_function(order, opt_einsum=False)
        expected_result = str(
            '\n'+
            'def solve_doubles_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Compute the change in the t_ij term (doubles)"""\n'+
            '\n'+
            '    if not trunc.doubles:\n'+
            '        raise Exception(\n'+
            '            "It appears that doubles is not true, this cannot be."\n'+
            '            "Something went terribly wrong!!!"\n'+
            '        )\n'+
            '    dt_ij = _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)\n'+
            '    return dt_ij\n'
        )
        assert function_output == expected_result

    def test_einsum(self):
        order = 2
        function_output = cdt._write_master_dt_amplitude_function(order, opt_einsum=True)
        expected_result = str(
            '\n'+
            'def solve_doubles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):\n'+
            '    """Compute the change in the t_ij term (doubles)"""\n'+
            '\n'+
            '    if not trunc.doubles:\n'+
            '        raise Exception(\n'+
            '            "It appears that doubles is not true, this cannot be."\n'+
            '            "Something went terribly wrong!!!"\n'+
            '        )\n'+
            '    dt_ij = _calculate_order_2_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list)\n'+
            '    return dt_ij\n'
        )
        assert function_output == expected_result


class Test_write_optimized_dt_amplitude_paths_function:

    def test_basic(self):
        max_order = 2
        function_output = cdt._write_optimized_dt_amplitude_paths_function(max_order)
        expected_result = str(
            '\n'+
            'def compute_optimized_paths(A, N, truncation):\n'+
            '    """Calculate optimized paths for the einsum calls up to `highest_order`."""\n'+
            '\n'+
            '    order_1_list, order_2_list, order_3_list = [], [], []\n'+
            '    order_4_list, order_5_list, order_6_list = [], [], []\n'+
            '\n'+
            '    return [None]\n'
        )
        assert function_output == expected_result


class Test_generate_dt_amplitude_string:

    def test_basic(self):
        max_order = 2
        function_output = cdt.generate_dt_amplitude_string(max_order, s1=75, s2=28)
        expected_result = str(  # file flag? or maybe cool
            '# --------------------------------------------------------------------------- #\n'+
            '# ---------------------------- DEFAULT FUNCTIONS ---------------------------- #\n'+
            '# --------------------------------------------------------------------------- #\n'+
            '\n'+
            '# ---------------------------- DISCONNECTED TERMS ---------------------------- #\n'+
            '\n'+
            'def _order_1_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n'+
            '        "which requires a residual of at least 2nd order"\n'+
            '    )\n'+
            '\n'+
            'def _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)\n'+
            '    But not terms (5), (3, 2), (2, 2, 1)\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'+
            '    # Creating the 2nd order return array\n'+
            '    linked_disconnected_terms = np.zeros((A, A, N, N), dtype=complex)\n'+
            '    # the (1, 1) term\n'+
            '    linked_disconnected_terms += 1/factorial(2) * (\n'+
            '        np.einsum(\'aci, cbj->abij\', dt_i, t_i) +\n'+
            '        np.einsum(\'aci, cbj->abij\', t_i, dt_i)\n'+
            '    )\n'+
            '\n'+
            '    return linked_disconnected_terms\n'+
            '\n'+
            'def _order_1_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'+
            '        "which requires a residual of at least 4th order"\n'+
            '    )\n'+
            '\n'+
            'def _order_2_un_linked_disconnected_terms(A, N, trunc, t_args, dt_args):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'+
            '        "which requires a residual of at least 4th order"\n'+
            '    )\n'+
            '\n'+
            '# ---------------------------- dt AMPLITUDES ---------------------------- #\n'+
            '\n'+
            'def _calculate_order_1_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Calculate the derivative of the 1 t-amplitude for use in the calculation of the residuals."""\n'+
            '    # unpack the `w_args`\n    w_i, *unusedargs = w_args\n    # Calculate the 1st order residual\n'+
            '    residual = residual_equations.calculate_order_1_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(1) * np.einsum(\'aci,cb->abi\', w_i, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        pass  # no linked disconnected terms for order < 2\n'+
            '    elif ansatz.VECC:\n'+
            '        pass  # no un-linked disconnected terms for order < 4\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_i = symmetrize_tensor(N, residual, order=1)\n'+
            '    return dt_i\n'+
            '\n'+
            'def _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals."""\n'+
            '    # unpack the `w_args`\n    w_i, w_ij, *unusedargs = w_args\n    # Calculate the 2nd order residual\n'+
            '    residual = residual_equations.calculate_order_2_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(2) * np.einsum(\'acij,cb->abij\', w_ij, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        residual -= _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n'+
            '    elif ansatz.VECC:\n'+
            '        residual -= _order_2_linked_disconnected_terms(A, N, trunc, t_args, dt_args)\n'+
            '        pass  # no un-linked disconnected terms for order < 4\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_ij = symmetrize_tensor(N, residual, order=2)\n'+
            '    return dt_ij\n'+
            '\n'+
            '# ---------------------------- WRAPPER FUNCTIONS ---------------------------- #\n'+
            '\n'+
            'def solve_singles_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Compute the change in the t_i term (singles)"""\n'+
            '\n'+
            '    if not trunc.singles:\n'+
            '        raise Exception(\n'+
            '            "It appears that singles is not true, this cannot be."\n'+
            '            "Something went terribly wrong!!!"\n'+
            '        )\n'+
            '    dt_i = _calculate_order_1_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)\n'+
            '    return dt_i\n'+
            '\n'+
            'def solve_doubles_equations(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args):\n'+
            '    """Compute the change in the t_ij term (doubles)"""\n'+
            '\n'+
            '    if not trunc.doubles:\n'+
            '        raise Exception(\n'+
            '            "It appears that doubles is not true, this cannot be."\n'+
            '            "Something went terribly wrong!!!"\n'+
            '        )\n'+
            '    dt_ij = _calculate_order_2_dt_amplitude(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args)\n'+
            '    return dt_ij\n'+
            '\n'+
            '# --------------------------------------------------------------------------- #\n'+
            '# --------------------------- OPTIMIZED FUNCTIONS --------------------------- #\n'+
            '# --------------------------------------------------------------------------- #\n'+
            '\n'+
            '# ---------------------------- DISCONNECTED TERMS ---------------------------- #\n'+
            '\n'+
            'def _order_1_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECI/CC term is (1, 1) or (dt_i * t_i)"\n'+
            '        "which requires a residual of at least 2nd order"\n'+
            '    )\n'+
            '\n'+
            'def _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Calculate all uniquely linked disconnected terms generated from the wave operator ansatz.\n'+
            '    This means for order 5 we include terms such as (4, 1), (3, 1, 1), (2, 1, 1, 1)\n'+
            '    But not terms (5), (3, 2), (2, 2, 1), (1, 1, 1, 1, 1)\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `t_args` and \'dt_args\'\n'+
            '    t_i, *unusedargs = t_args\n'+
            '    dt_i, *unusedargs = dt_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'+
            '    # Creating the 2nd order return array\n'+
            '    linked_disconnected_terms = np.zeros((A, A, N, N), dtype=complex)\n'+
            '    # the (1, 1) term\n'+
            '    linked_disconnected_terms += 1/factorial(2) * (\n'+
            '        next(optimized_einsum)(dt_i, t_i) +\n'+
            '        next(optimized_einsum)(t_i, dt_i)\n'+
            '    )\n'+
            '\n'+
            '    return linked_disconnected_terms\n'+
            '\n'+
            'def _order_1_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'+
            '        "which requires a residual of at least 4th order"\n'+
            '    )\n'+
            '\n'+
            'def _order_2_un_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list):\n'+
            '    """Exists for error checking."""\n'+
            '    raise Exception(\n'+
            '        "the first possible purely VECC term is (2, 2) or (dt_ij * t_ij)"\n'+
            '        "which requires a residual of at least 4th order"\n'+
            '    )\n'+
            '\n'+
            '# ---------------------------- dt AMPLITUDES ---------------------------- #\n'+
            '\n'+
            'def _calculate_order_1_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):\n'+
            '    """Calculate the derivative of the 1 t-amplitude for use in the calculation of the residuals.\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `w_args`\n'+
            '    w_i, *unusedargs = w_args\n'+
            '    # Calculate the 1st order residual\n'+
            '    residual = residual_equations.calculate_order_1_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(1) * opt_epsilon(w_i, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        pass  # no linked disconnected terms for order < 2\n'+
            '    elif ansatz.VECC:\n'+
            '        pass  # no un-linked disconnected terms for order < 4\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_i = symmetrize_tensor(N, residual, order=1)\n'+
            '    return dt_i\n'+
            '\n'+
            'def _calculate_order_2_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_epsilon, opt_path_list):\n'+
            '    """Calculate the derivative of the 2 t-amplitude for use in the calculation of the residuals.\n'+
            '    Uses optimized summation paths generate using `contract_expression` from the `opt_einsum` library.\n'+
            '    """\n'+
            '    # unpack the `w_args`\n'+
            '    w_i, w_ij, *unusedargs = w_args\n'+
            '    # make an iterable out of the `opt_path_list`\n'+
            '    optimized_einsum = iter(opt_path_list)\n'+
            '    # Calculate the 2nd order residual\n'+
            '    residual = residual_equations.calculate_order_2_residual(A, N, trunc, h_args, w_args)\n'+
            '    # subtract the epsilon term (which is R_0)\n'+
            '    residual -= 1/factorial(2) * opt_epsilon(w_ij, epsilon)\n'+
            '\n'+
            '    # subtract the disconnected terms\n'+
            '    if ansatz.VECI:\n'+
            '        pass  # veci does not include any disconnected terms\n'+
            '    elif ansatz.VE_MIXED:\n'+
            '        residual -= _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n'+
            '    elif ansatz.VECC:\n'+
            '        residual -= _order_2_linked_disconnected_terms_optimized(A, N, trunc, t_args, dt_args, opt_path_list)\n'+
            '        pass  # no un-linked disconnected terms for order < 4\n'+
            '\n'+
            '    # Symmetrize the residual operator\n'+
            '    dt_ij = symmetrize_tensor(N, residual, order=2)\n'+
            '    return dt_ij\n'+
            '\n'+
            '# ---------------------------- WRAPPER FUNCTIONS ---------------------------- #\n'+
            '\n'+
            'def solve_singles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):\n'+
            '    """Compute the change in the t_i term (singles)"""\n'+
            '\n'+
            '    if not trunc.singles:\n'+
            '        raise Exception(\n'+
            '            "It appears that singles is not true, this cannot be."\n'+
            '            "Something went terribly wrong!!!"\n'+
            '        )\n'+
            '    dt_i = _calculate_order_1_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list)\n'+
            '    return dt_i\n'+
            '\n'+
            'def solve_doubles_equations_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list):\n'+
            '    """Compute the change in the t_ij term (doubles)"""\n'+
            '\n'+
            '    if not trunc.doubles:\n'+
            '        raise Exception(\n'+
            '            "It appears that doubles is not true, this cannot be."\n'+
            '            "Something went terribly wrong!!!"\n'+
            '        )\n'+
            '    dt_ij = _calculate_order_2_dt_amplitude_optimized(A, N, ansatz, trunc, epsilon, h_args, t_args, dt_args, w_args, opt_path_list)\n'+
            '    return dt_ij\n'+
            '\n'+
            '# ---------------------------- OPTIMIZED PATHS FUNCTION ---------------------------- #\n'+
            '\n'+
            'def compute_optimized_paths(A, N, truncation):\n'+
            '    """Calculate optimized paths for the einsum calls up to `highest_order`."""\n'+
            '\n'+
            '    order_1_list, order_2_list, order_3_list = [], [], []\n'+
            '    order_4_list, order_5_list, order_6_list = [], [], []\n'+
            '\n'+
            '    return [None]\n'+
            '\n'
        )
        assert function_output == expected_result


class Test_generate_dt_amplitude_equations_file:

    def test_run_main(self):
        cdt.generate_dt_amplitude_equations_file(2, path="./dt_amplitude_equations.py")