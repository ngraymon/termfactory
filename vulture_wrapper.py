import sys
import os

if (__name__ == '__main__'):

    sys.stdout = open('vulture_output.txt', 'w')

    os.system(
        "py -m vulture --min-confidence 100 generate_residual_equations.py"
    )
    sys.stdout.close()
