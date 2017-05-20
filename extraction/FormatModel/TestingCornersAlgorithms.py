import numpy as np
import cv2
import time
import helloworld
def countNonZero(sum, i_j, x_y = None):
    if x_y is None:
        i = i_j[0]
        j = i_j[1]
        if i<0 or j<0:
            return 0

        return sum[i,j]
    else:
        i = i_j[0]
        j = i_j[1]
        x = x_y[0]
        y = x_y[1]
        T = countNonZero(sum, i_j=x_y)
        A = countNonZero(sum, i_j=(i-1,j-1))
        P = countNonZero(sum, i_j=(x, j-1))
        Q = countNonZero(sum, i_j=(i-1, y))
        return T-P-Q+A

def createSum(A):
    sum = np.zeros(A.shape)
    rows, cols = A.shape
    for x in range(rows):
        for y in range(cols):
            T = countNonZero(sum, i_j=(x-1, y - 1))
            P = countNonZero(sum, i_j=(x - 1, y))
            Q = countNonZero(sum, i_j=(x, y - 1))
            S = P + Q - T
            if A[x,y] != 0:
                S += 1
            sum[x,y] = S
    return sum

if __name__ == '__main__':
    A = np.zeros((4,3))
    A[0, 1] = 1
    A[1, 2] = 1
    A[3, 2] = 1
    A[2, 0] = 1
    print(A)
    S = createSum(A)
    print(S)
    start_time = time.time()
    A = cv2.imread('/home/vmchura/Documents/handwritten/input/pagina1_1.png', 0)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    S = createSum(A)
    print("--- %s seconds ---" % (time.time() - start_time))