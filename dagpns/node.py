from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class Node:

    def __init__(self, code,phi, delta, parents=None):
        self.delta=delta
        self.phi=phi
        self.code=code
        self.parents=parents


if __name__ == "__main__":
    print("hello")
    a=[3,4,5,6,7]
    b=[1]
    x= [a[i] for i in range(len(a)) if i%2==0]
    print(x)
    import numpy as np
    A=np.arange(10)
    print(A)
    print(A[b])