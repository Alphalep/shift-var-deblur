import numpy as np
img = np.resize([i for i in range(1,17)],(4,4))
print(img.shape[0:2])
kernel = np.array([[2,0],
             [0,2]])
k = 2
def unfold_matrix(X, k):
    n, m = X.shape[0:2]
    xx = np.zeros(((n - k + 1) * (m - k + 1),k**2))
    xx2 = np.zeros(((n - k + 1) * (m - k + 1),k**2))
    row_num = 0
    def make_row(x):
        return x.flatten()

    for i in range(n- k+ 1):
        for j in range(m - k + 1):
            #collect block of m*m elements and convert to row
            xx[row_num,:] = X[i:i+k, j:j+k].flatten()
            xx2[row_num,:] = make_row(X[i:i+k, j:j+k])
            row_num = row_num + 1


    if (xx == xx2).all():
        print('Oh bazigar Oh Bazigarrrr')
    return xx

print(unfold_matrix(img,k))