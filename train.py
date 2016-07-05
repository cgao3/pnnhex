

import numpy as np

def ack(m, n):
    A=np.zeros(dtype=np.int32, shape=(m+1, n+100))

    for j in range(n+100):
        A[0][j]=j+1

    for i in range(1, m+1):
        for j in range(n+50):
            if j==0:
                A[i][j]=A[i-1][1]
            else:
                A[i][j]=A[i-1][A[i][j-1]]

    print(A)
    print("ack(%d,%d)"%(m,n), "=", A[m][n])

ack(3,4)
