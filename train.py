import tensorflow as tf

def f(x):
    return x**2

def calculus(func, a, b):
    epsilon=0.001
    width=abs(a-b)
    seg=width/epsilon
    area=0.0
    for i in range(int(seg)):
        subarea=epsilon*(f(a)+f(b))/2
        area += subarea

    return area

print("area is:", calculus(f,6,9))