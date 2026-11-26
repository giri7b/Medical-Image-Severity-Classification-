import pathlib

p = pathlib.Path('preprocessing.py')
lines = p.read_text('utf-8').splitlines()
for i, line in enumerate(lines, start=1):
    if '→' in line:
        print('Found arrow on line', i, repr(line))
