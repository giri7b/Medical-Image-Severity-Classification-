import pathlib

p = pathlib.Path('preprocessing.py')
lines = p.read_text('utf-8').splitlines()
line = lines[351]
print('LINE 352:', repr(line))
print('CODEPOINTS:', [hex(ord(c)) for c in line])
