# termfactory
[![Master_workflow_coverage](https://github.com/ngraymon/termfactory/actions/workflows/master_coverage.yml/badge.svg)](https://pypi.org/project/termfactory/)
[![Testing_workflow_coverage](https://github.com/ngraymon/termfactory/actions/workflows/testing_coverage.yml/badge.svg?testing_coverage=ci_testing)](https://pypi.org/project/termfactory/)
[![codecov](https://img.shields.io/codecov/c/github/ngraymon/termfactory/master.svg?label=master)](https://codecov.io/gh/ngraymon/termfactory)
[![codecov](https://img.shields.io/codecov/c/github/ngraymon/termfactory/ci_testing.svglabel=ci_testing)](https://codecov.io/gh/ngraymon/termfactory)

----
## About

A python package for generating LaTeX and python code for evaluating residual terms.
The package is used to generate the `einsum` equations for the [vecc package (link tbd)]().


At the moment the package provides three primary ansatz:

- full CC
      `(H, CC, P)`
- HZ ansatz `(H, CC, S, P)`
- e^T HZ
     `(H, CC, T, eT, P)`

Both LaTeX and python code can be generated.
Ground state equations are available for all three ansatz.
There are no excited state equations available for HZ ansatz.

## Install
`pip install termfactory`


## Usage
driver.py is the main argparse interface `py driver.py -h`

```shell
usage: driver.py [-h] [-l /path/filename.txt] [-s] [-t T [T ...]] [-a ANSATZ] [-es EXCITED_STATE] [-rf REMOVE_F_TERMS] [-c] [-lhs] [-p PATH]

Code/Latex Generator

optional arguments:
  -h, --help            show this help message and exit
  -l /path/filename.txt, --log_path /path/filename.txt
                        path to log file (default: default_logging_file.txt)
  -s, --stdlog          provide if you want the log to be piped to stdout (default: False)
  -t T [T ...]          Provided Truncations, example: -t 2 2 2 2 (default: None)
  -a ANSATZ, --ansatz ANSATZ
                        Specify Ansatz (default: None)
  -es EXCITED_STATE, --excited_state EXCITED_STATE
                        Only ground state? (default: True)
  -rf REMOVE_F_TERMS, --remove_f_terms REMOVE_F_TERMS
                        Choose to remove f terms (default: False)
  -c, --code            Generate LaTeX by default; `-c` generates code instead. (default: False)
  -lhs                  Generate LaTeX by default; `-c` generates code instead. (default: False)
  -p PATH, --path PATH  filename of load/save file (default: None)
```

## Examples

Every execution produces three primary files:

- `filename.(py/tex)`
- `default_logging_file.txt`
- `truncs.json`


### Generating LaTeX

#### full CC
----

##### RHS
`python3 driver.py -t 1 1 1 1`

Default filename: `ground_state_full_cc_symmetric_equations.tex`


##### LHS
`python3 driver.py -t 1 1 1 1 -lhs`

Default filename: `ground_state_full_cc_special_LHS_terms.tex`




#### HZ ansatz
----


##### RHS
`python3 driver.py -t 1 1 1 1`

Default filename: `ground_state_z_t_symmetric_equations.tex`

##### LHS
`python3 driver.py -t 1 1 1 1 -lhs`

Default filename: `ground_state_z_t_special_LHS_terms.tex`



#### e^T HZ
----

##### RHS
`python3 driver.py -t 1 1 1 1 1`

Default filename: `ground_state_eT_z_t_symmetric_equations.tex`

##### LHS
`python3 driver.py -t 1 1 1 1 1 -lhs`

Default filename: `ground_state_eT_z_t_special_LHS_terms.tex`



### Generating Python code


#### full CC
----

##### RHS
`python3 driver.py -t 1 1 1 1 -c`

Default filename: `ground_state_full_cc_equations.py`

##### LHS
`python3 driver.py -t 1 1 1 1 1 -c -lhs`

Default filename: `ground_state_full_cc_special_LHS_equations.py`



#### HZ ansatz
----
There is no code generator implemented for the HZ ansatz


#### e^T HZ
----

##### RHS
`python3 driver.py -t 1 1 1 1 1 -c`

Default filename: `eT_zhz_eqs_RHS_H(1)_P(1)_T(1)_exp(1)_Z(1).py`

##### LHS
`python3 driver.py -t 1 1 1 1 1 -c -lhs`

Default filename: `eT_zhz_eqs_LHS_H(1)_P(1)_T(1)_exp(1)_Z(1).py`






