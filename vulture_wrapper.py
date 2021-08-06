import subprocess
import sys

if (__name__ == '__main__'):

    # sys.stdout = open('vulture_output.txt', 'w')
    # sys.stderr = sys.stdout

    # print("testing stdout")

    ret = subprocess.run(
        ['py', '-m', 'vulture', '--min-confidence 100', 'generate_residual_equations.py'],
        capture_output=True,
    )

    with open('vulture_output.txt', 'w') as fp:
        fp.write(ret.stdout)
        fp.write(ret.stderr)

    # sys.stdout.close()
