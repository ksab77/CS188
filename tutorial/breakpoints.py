import pdb

def foo():
    a = 5
    b = [7, 8, 9]
    print a * b

def bar():
    a = 3
    b = "cool "
    pdb.set_trace()
    foo()
    print a * b
