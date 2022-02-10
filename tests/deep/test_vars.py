from logging import root
from . import context
import latex_full_cc as fcc
import code_residual_equations as cre


class write_cc_einsum_python_from_list_single_unique_key:

    t_term_list = [
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0], n_t=[1]),
            fcc.connected_h_operator_namedtuple(rank=1, m=1, n=0, m_o=1, n_o=0, m_t=[0], n_t=[0]),
            (fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),)
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=0, m_t=[0, 0], n_t=[1, 1]),
            fcc.connected_h_operator_namedtuple(rank=0, m=0, n=0, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 0]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=0, m_t=[0, 0], n_t=[1, 1]),
            fcc.connected_h_operator_namedtuple(rank=1, m=0, n=1, m_o=0, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=1, n_o=0)
            )
        ],
        [
            fcc.connected_omega_operator_namedtuple(rank=2, m=0, n=2, m_h=0, n_h=1, m_t=[0, 0], n_t=[1, 0]),
            fcc.connected_h_operator_namedtuple(rank=2, m=1, n=1, m_o=1, n_o=0, m_t=[0, 0], n_t=[0, 1]),
            (
                fcc.disconnected_namedtuple(m_h=0, n_h=0, m_o=1, n_o=0),
                fcc.connected_namedtuple(m_h=1, n_h=0, m_o=0, n_o=0)
            )
        ]
    ]

    output = [
        'if truncation.singles:',
        '    R += 1/(factorial(2)) * (',
        "        np.einsum('ac, cdz, dby -> abzy', h_args[(0, 0)], t_args[(1, 0)], t_args[(1, 0)]) +",
        "        np.einsum('ac, cdy, dbz -> abzy', h_args[(0, 0)], t_args[(1, 0)], t_args[(1, 0)])",
        '    )',
        '',
        'if truncation.at_least_linear:',
        '    if truncation.singles:',
        '        R += (',
        "            np.einsum('acz, cby -> abzy', h_args[(1, 0)], t_args[(1, 0)]) +",
        "            np.einsum('acz, cby -> abzy', h_args[(1, 0)], t_args[(1, 0)])",
        '        )',
        '        R += 1/(factorial(2)) * (',
        "            np.einsum('aciz, cdy, dbi -> abzy', h_args[(1, 1)], t_args[(1, 0)], t_args[(1, 0)]) +",
        "            np.einsum('aciz, cdi, dby -> abzy', h_args[(1, 1)], t_args[(1, 0)], t_args[(1, 0)])",
        '        )',
        '    if truncation.doubles:',
        '        R += 1/(factorial(2)) * (',
        "            np.einsum('aci, cdz, dbiy -> abzy', h_args[(0, 1)], t_args[(1, 0)], t_args[(2, 0)]) +",
        "            np.einsum('aci, cdiy, dbz -> abzy', h_args[(0, 1)], t_args[(2, 0)], t_args[(1, 0)])",
        '        )',
        '',
        'if truncation.at_least_quadratic:'
    ]


class generate_residual_data:

    expected_result = (
        [
            ['h_0', 'h_{k_1} * w^{k_1}'],
            ['h_0 * w^{i_1}', 'h_{k_1} * w^{i_1,k_1}', 'h^{i_1}']
        ],
        [
            [
                cre.residual_term(
                    prefactor='1.0',
                    h=cre.h_namedtuple(max_i=0, max_k=0),
                    w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
                ),
                cre.residual_term(
                    prefactor='1.0',
                    h=cre.h_namedtuple(max_i=0, max_k=1), 
                    w=cre.w_namedtuple(max_i=0, max_k=1, order=1)
                )
            ],
            [
                cre.residual_term(
                    prefactor='1.0',
                    h=cre.h_namedtuple(max_i=0, max_k=0),
                    w=cre.w_namedtuple(max_i=1, max_k=0, order=1)
                ),
                cre.residual_term(
                    prefactor='1.0',
                    h=cre.h_namedtuple(max_i=0, max_k=1),
                    w=cre.w_namedtuple(max_i=1, max_k=1, order=2)
                ),
                cre.residual_term(
                    prefactor='1.0',
                    h=cre.h_namedtuple(max_i=1, max_k=0),
                    w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
                )
            ]
        ]
    )


class write_residual_function_string_high_order_h_and_w:

    res_list = [
        cre.residual_term(
            prefactor='(1/2)',
            h=cre.h_namedtuple(max_i=0, max_k=0),
            w=cre.w_namedtuple(max_i=2, max_k=0, order=2)
        ),
        cre.residual_term(
            prefactor='(3/6)',
            h=cre.h_namedtuple(max_i=0, max_k=1),
            w=cre.w_namedtuple(max_i=2, max_k=1, order=3)
        ),
        cre.residual_term(
            prefactor='(6/24)',
            h=cre.h_namedtuple(max_i=0, max_k=2),
            w=cre.w_namedtuple(max_i=2, max_k=2, order=4)
        ),
        cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=1, max_k=0),
            w=cre.w_namedtuple(max_i=1, max_k=0, order=1)
        ),
        cre.residual_term(
            prefactor='1.0',
            h=cre.h_namedtuple(max_i=1, max_k=1),
            w=cre.w_namedtuple(max_i=1, max_k=1, order=2)
        ),
        cre.residual_term(
            prefactor='(1/2)',
            h=cre.h_namedtuple(max_i=2, max_k=0),
            w=cre.w_namedtuple(max_i=0, max_k=0, order=0)
        )
    ]
