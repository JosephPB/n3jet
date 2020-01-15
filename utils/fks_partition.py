import numpy as np

def dot(p1,p2):
    'Minkowski metric dot product'
    prod = p1[0]*p2[0]-(p1[1]*p2[1]+p1[2]*p2[2]+p1[3]*p2[3])
    return prod

def s(p_1,p_2):
    'CoM energy of two massless jets'
    return (2*dot(p_1,p_2))

def d_ij(mom,i,j):
    'CoM energy of selected massless jets'
    return s(mom[i],mom[j])

def D_ij(mom,n_gluon):
    'Reciprocal of CoM energy pairwise sum'
    ds = []
    pairs = []
    for i in range(2,n_gluon+2+2):
        for j in range(i+1,n_gluon+2+2):
            ds.append(d_ij(mom,i,j))
            pairs.append([i,j])
    return np.sum(1/np.array(ds)), pairs

def S_ij(mom,n_gluon,i,j):
    'Partition function'
    D_1,_ = D_ij(mom,n_gluon)
    return (1/(D_1*d_ij(mom,i,j)))