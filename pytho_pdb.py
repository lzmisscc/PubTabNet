
def q(x):
    if x <= 1:
        return x
    while x:
        return x*(x-1)
for ii in range(10000):
    q(1e+100)