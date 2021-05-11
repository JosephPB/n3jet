import numpy as np
from tqdm import tqdm

from n3jet.utils.general_utils import dot
from n3jet.phase import check_all

class FKSPartition:

    def __init__(
            self,
            momenta,
            labels,
            all_legs = False,
    ):
        
        self.momenta = momenta
        self.labels = labels
        self.all_legs = all_legs

        if type(self.momenta) != list:
            raise AssertionError('Momentum must be in the form of a list')
        
    def cut_near_split(self, delta_cut, delta_near, return_indices = False):
        '''
        Split momenta into near and cut arrays - 
          near is the region close to the PS cuts and the cut region is the rest of the cut PS

        :param delta_cut: the PS cut delta
        :param delta_near: the secondary 'cut' defining the region 'close to' the cut boundary
        '''

        cut_momenta = []
        self.near_momenta = []
        cut_labels = []
        self.near_labels = []
        cut_indices = []
        near_indices = []

        for idx, i in tqdm(enumerate(self.momenta), total = len(self.momenta)):
            close, min_distance = check_all(
                mom=i,
                delta=delta_cut,
                s_com=dot(i[0],i[1]),
                all_legs=self.all_legs
            )
            if not close:
                if min_distance < delta_near:
                    self.near_momenta.append(i)
                    self.near_labels.append(self.labels[idx])
                    near_indices.append(idx)
                else:
                    cut_momenta.append(i)
                    cut_labels.append(self.labels[idx])
                    cut_indices.append(idx)

        return cut_momenta, self.near_momenta, cut_labels, self.near_labels, cut_indices, near_indices

    def s(self, p_1,p_2):
        'CoM energy of two massless jets'
        return (2*dot(p_1,p_2))

    def d_ij(self, mom, i,j):
        'CoM energy of selected massless jets'
        return self.s(mom[i],mom[j])

    def D_ij(self, mom):
        "Reciprocal of CoM energy for pairwise sum"
        ds = []
        pairs = []

        if not self.all_legs:
            for i in range(2, len(mom)):
                for j in range(i+1, len(mom)):
                    ds.append(self.d_ij(mom,i,j))
                    pairs.append([i,j])

        else:
            for i in range(len(mom)):
                for j in range(i+1, len(mom)):
                    if i == 0 and j == 1:
                        pass
                    else:
                        ds.append(self.d_ij(mom,i,j))
                        pairs.append([i,j])
                        
        return np.sum(1/np.array(ds)), pairs

    def S_ij(self, mom, i, j):
        'Partition function'
        D_1,_ = self.D_ij(mom)
        return (1/(D_1*self.d_ij(mom,i,j)))

    def weighting(self, return_weights = False):
        '''
        Weights scattering amplitudes according to the different partition function for pairs of particle
        '''

        D_1, pairs = self.D_ij(self.near_momenta[0])
        S_near = []

        for idx, i in enumerate(pairs):
            print ('Pair {} of {}'.format(idx+1, len(pairs)))
            S = []
            for j in tqdm(self.near_momenta, total=len(self.near_momenta)):
                S.append(self.S_ij(j,i[0],i[1]))
            S_near.append(np.array(S))
        S_near = np.array(S_near)

        labs_split = []
        for i in S_near:
            labs_split.append(self.near_labels*i)

        if return_weights:
            return pairs, labs_split, S_near

        else:
            return pairs, labs_split
