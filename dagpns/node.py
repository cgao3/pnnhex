from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class Node:

    def __init__(self, theta, phi):
        self.theta=theta
        self.phi=phi

    def expand(self):

        pass


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