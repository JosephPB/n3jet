import numpy as np
from tqdm import tqdm
import pandas as pd

# set com enery
s = 500

# set ps variable product
delta_vars = 0.2

def dot(p1,p2):
    'Minkowski metric dot product'
    prod = p1[0]*p2[0]-(p1[1]*p2[1]+p1[2]*p2[2]+p1[3]*p2[3])
    return prod

def pair_check(p1,p2,delta,s_com):
    '''Check proximity of pair of momenta
    :param p1, p2: 4-momenta
    :param delta: proximity measure according to the JADE algorithm - e.g. 0.01
    
    returns: boolean - True if too close, False otherwise
    '''
    
    distance = (dot(p1,p2))/s_com
        
    close = False
    if distance <= delta:
        close = True
        
    return close, distance

def check_all(mom,delta,s_com, all_legs=False):
    'Given an array of 4-momenta, check proximity of all pairs'
    too_close = False

    if not all_legs:
        p_array = mom[2:]
    else:
        p_array = mom

    distances = []
    for idx, p in enumerate(p_array):
        to_check = p_array[idx+1:]
        for j in to_check:
            close, distance = pair_check(p,j,delta,s_com=s_com)
            distances.append(distance)
            if close == True:
                too_close = True
                
    return too_close, np.sort(distances)[0]

class Rambo:
    def __init__(self, num_jets, num_points, w, delta_cut, all_legs=False):
        self.num_jets = num_jets
        self.num_points = num_points
        self.w = w
        self.delta_cut = delta_cut
        self.all_legs = all_legs


    def random_moms(self):
        'Generate 4-mom components from a uniform distribution'
        moms = []
        for i in range(self.num_jets):
            moms.append(np.random.uniform(0,1,4))
        return np.array(moms)

    def isotropic_moms(self, mom):
        'Create massles 4-mom with isotropic angular distribution'
        c = 2*mom[0] - 1
        phi = 2*np.pi*mom[1]
        q_0 = -np.log(mom[2]*mom[3])
        q_1 = q_0*np.sqrt(1-c**2)*np.cos(phi)
        q_2 = q_0*np.sqrt(1-c**2)*np.sin(phi)
        q_3 = q_0*c
        return np.array([q_0,q_1,q_2,q_3])

    def boost(self, mom, moms):
        '''
        Boost and scale isotropic 4-mom for momentum conservation
        :param mom: one isotropic 4-mom
        :param moms: array of all isotropic momenta
        :param w: centre of mass energy

        returns: np array of boosted an scaled 4-mom corresponding to mom
        '''
        q = mom
        Q = np.zeros(4)
        for i in moms:
            Q+=i
        M = np.sqrt(dot(Q,Q))
        b = -Q[1:]/M
        x = self.w/M
        gamma = Q[0]/M
        a = 1/(1+gamma)

        p_0 = x*(gamma*q[0]+np.dot(b,q[1:]))
        p_space = x*(q[1:]+b*q[0]+a*(np.dot(b,q[1:]))*b)
        return np.array([p_0,p_space[0],p_space[1],p_space[2]])


    def generate(self):

        p_1 = np.array([self.w/2,0.,0.,self.w/2])
        p_2 = np.array([self.w/2,0.,0.,-self.w/2])

        cut_momenta = []

        pbar = tqdm(total=self.num_points)
        while len(cut_momenta) < self.num_points:
            moms = self.random_moms()

            iso_moms = []
            for i in moms:
                iso_moms.append(self.isotropic_moms(i))

            iso_moms = np.array(iso_moms)

            boost_moms = []
            boost_moms.append(p_1)
            boost_moms.append(p_2)
            for i in iso_moms:
                boost_moms.append(self.boost(i,iso_moms))

            close, _ = check_all(
                mom=boost_moms,
                delta=self.delta_cut,
                s_com=dot(p_1,p_2)
            )

            if close == False:
                cut_momenta.append(boost_moms)
                pbar.update(1)

        pbar.close()
        cut_mom = pd.DataFrame({'momenta':list(cut_momenta)})

        return cut_mom['momenta']

    
    def generate_piecewise(self, delta_near=0.02):

        p_1 = np.array([self.w/2,0.,0.,self.w/2])
        p_2 = np.array([self.w/2,0.,0.,-self.w/2])

        count_near = 0
        count_cut = 0
        cut_momenta = []
        near_momenta = []

        pbar = tqdm(total=self.num_points)
        distance = []
        while (count_near + count_cut) < self.num_points:
            moms = self.random_moms()

            iso_moms = []
            for i in moms:
                iso_moms.append(self.isotropic_moms(i))

            iso_moms = np.array(iso_moms)

            boost_moms = []
            boost_moms.append(p_1)
            boost_moms.append(p_2)
            for i in iso_moms:
                boost_moms.append(self.boost(i,iso_moms))

            close, min_distance = check_all(
                mom=boost_moms,
                delta=self.delta_cut,
                s_com=dot(p_1,p_2),
                all_legs=self.all_legs
            )

            if close == False:
                if min_distance < delta_near:
                    if count_near < (1*self.num_points)/2:
                        near_momenta.append(boost_moms)
                        pbar.update(1)
                        count_near += 1
                else:
                    if count_cut < (1*self.num_points)/2:
                        cut_momenta.append(boost_moms)
                        pbar.update(1)
                        count_cut += 1

        pbar.close()
        
        cut_mom = pd.DataFrame({'momenta':list(cut_momenta)})
        near_mom = pd.DataFrame({'momenta':list(near_momenta)})

        return cut_mom['momenta'], near_mom['momenta']
