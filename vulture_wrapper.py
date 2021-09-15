import subprocess

if (__name__ == '__main__'):

    ret = subprocess.run(
        ['python', '-m', 'vulture', 'driver.py'],
        capture_output=True,
        text=True,
    )

    with open('vulture_output.txt', 'w') as fp:
        fp.write(ret.stdout)
        fp.write(ret.stderr)
