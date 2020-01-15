import numpy as np
from tqdm import tqdm
import pandas as pd

# set com enery
s = 1000**2

# set 4-momentum proximity by angle
delta_angle = np.cos(0.4)
# set ps variable product
delta_vars = 0.2

def pair_check(p1,p2,delta=delta_angle):
    '''Check proximity of pair of momenta
    :param p1, p2: 4-momenta
    :param delta: proximity measure - e.g. np.cos(0.4)
    returns: boolean - True if too close, False otherwise
    '''
    p1 = p1[1:]
    p2 = p2[1:]
    
    cos_angle = (np.dot(p1,p2))/((np.sqrt(np.dot(p1,p1))*np.sqrt(np.dot(p2,p2))))
    #if cos_angle < 0:
    #    cos_angle = -cos_angle
        
    close = False
    if cos_angle >= delta:
        close = True
    
    return close

def check_all(p_array):
    'Given an array of 4-momenta, check proximity of all paira'
    too_close = False
    
    for idx, p in enumerate(p_array):
        to_check = p_array[idx+1:]
        for j in to_check:
            proximity = pair_check(p,j)
            if proximity == True:
                too_close = True
                
    return too_close 

def out_particles_2_2(theta, phi=0):
    p_3 = (np.sqrt(s)/2)*np.array([1.,np.sin(theta)*np.cos(phi),-np.sin(phi)*np.sin(theta),np.cos(theta)])
    p_4 = (np.sqrt(s)/2)*np.array([1.,-np.sin(theta)*np.cos(phi),np.sin(phi)*np.sin(theta),-np.cos(theta)])
    return p_3, p_4

def is_soft(variables,delta=delta_vars):
    variables = np.array(variables)
    prod = np.prod(variables)
    if prod < delta:
        return True
    else:
        return False

def ps_2_2(num_points):
    '''
    Generates num_points phase space points for 2->2
    returns: momenta, theta after proximity checking
    
    note: len(cut_mom) <= num_points due to proximity checking
    '''
    
    p_1 = (np.sqrt(s)/2)*np.array([1.,0.,0.,1.])
    p_2 = (np.sqrt(s)/2)*np.array([1.,0.,0.,-1.])
    
    theta_range = np.random.uniform(0,np.pi,num_points)
    
    momenta = []
    for i in range(len(theta_range)):
        p_3, p_4 = out_particles_2_2(theta_range[i])
        momenta.append([list(p_1),list(p_2),list(p_3),list(p_4)])
        
    close = []
    for i in tqdm(momenta):
        close.append(check_all(i))
        
    data = {'momenta':list(momenta), 'theta':list(theta_range), 'too close':list(close)}
    
    df_2_2 = pd.DataFrame(data)
    cut_2_2 = df_2_2[df_2_2['too close'] == False].reset_index()
    cut_mom = cut_2_2['momenta']
    cut_theta = cut_2_2['theta']
    
    return cut_mom, cut_theta
    
def out_particles_2_3(theta,alpha,x_1,x_2, phi = 0):
        
    p_3 = x_1*(np.sqrt(s)/2)*np.array([1.,np.sin(theta)*np.cos(phi),-np.sin(phi)*np.sin(theta),np.cos(theta)])
    
    root = np.sqrt(((x_1-1)*(x_2-1)*(x_1+x_2-1))/((x_1**2)*(x_2**2)))

    p_4 = x_2*(np.sqrt(s)/2)*np.array([1.,(2*x_1*x_2*root*np.cos(alpha)*np.cos(theta)*np.cos(phi)+(x_1*(x_2-2)-2*x_2+2)*np.sin(theta)*np.cos(phi)-2*x_1*x_2*root*np.sin(alpha)*np.sin(phi))/(x_1*x_2),-(2*x_1*x_2*root*np.cos(phi)*np.sin(alpha)+(2*x_1*x_2*root*np.cos(alpha)*np.cos(theta)+(x_1*(x_2-2)-2*x_2+2)*np.sin(theta))*np.sin(phi))/(x_1*x_2),((x_1*(x_2-2)-2*x_2+2)*np.cos(theta)-2*x_1*x_2*root*np.cos(alpha)*np.sin(theta))/(x_1*x_2)])
    
    p_5 = np.array([(2-x_1-x_2)*(np.sqrt(s)/2),-np.sqrt(s)*(2*x_1*x_2*root*np.cos(alpha)*np.cos(theta)*np.cos(phi)+(x_1**2+(x_2-2)*x_1-2*x_2+2)*np.sin(theta)*np.cos(phi)-2*x_1*x_2*root*np.sin(alpha)*np.sin(phi))/(2*x_1),np.sqrt(s)*(2*x_1*x_2*root*np.cos(phi)*np.sin(alpha)+(2*x_1*x_2*root*np.cos(alpha)*np.cos(theta)+(x_1**2+(x_2-2)*x_1-2*x_2+2)*np.sin(theta))*np.sin(phi))/(2*x_1),np.sqrt(s)*(2*x_1*x_2*root*np.cos(alpha)*np.sin(theta)-(x_1**2+(x_2-2)*x_1-2*x_2+2)*np.cos(theta))/(2*x_1)])
    
    return p_3.astype(np.float64).tolist(), p_4.astype(np.float64).tolist(), p_5.astype(np.float64).flatten().tolist()
    
def ps_2_3(num_points, vary, fix):
    '''
    Generates num_points phase space points for 2->3
    :param vary: array of strings of variables to vary, e.g. ['theta',  'alpha']
    :param fix: array of tubles of variables to fix, e.g. [('theta',0.5)]
    
    returns: points, theta, alpha, x_1, x_2, x_3 after proximimty checking
    
    note: require minimum of 'theta', 'alpha' and 'x_1' to be in either vary or fix
    note: always specify 'x_1' before 'x_2'
    note: if 'x_1' is fixed, but 'x_2' is not specified then assumed to vary
    '''
    
    p_1 = (np.sqrt(s)/2)*np.array([1.,0.,0.,1.])
    p_2 = (np.sqrt(s)/2)*np.array([1.,0.,0.,-1.])

    to_set = ['theta', 'alpha', 'x_1']
    
    setting = []
    varying = []
    fixing = []
    for i in vary:
        setting.append(i)
        varying.append(i)
    for i in fix:
        setting.append(i[0])
        fixing.append(i[0])
        
    for i in to_set:
        if i not in setting:
            raise Exception('Error: {} not set'.format(i))
    
    
    for i in vary:
        if i == 'theta':
            theta_range = np.random.uniform(0, np.pi, num_points)
        if i == 'alpha':
            alpha_range = np.random.uniform(-np.pi, np.pi, num_points)
        if i == 'x_1':
            # all x are dependend on each other
            # for now we do not allow for fixing of one x_i
            # TODO: assess if individual x_i fixing necessary
            x_1_range = np.random.uniform(0,1,10000)
            x_1sub = 1-x_1_range
            
            x_2_range = []
            for i in x_1sub:
                x_2 = np.random.uniform(i,1,1)
                x_2_range.append(x_2[0])
            x_2_range = np.array(x_2_range)
            
            x_3_range = 2-x_1_range-x_2_range
            
    for i in fix:
        if i[0] == 'theta':
            if i[1] < 0 or i[1] > np.pi:
                raise Exception('Error: theta should be in range 0<theta<pi')
            theta_range = np.ones(num_points)*i[1]
        if i[0] == 'alpha':
            if i[1] < -np.pi or i[1] > np.pi:
                raise Exception('Error: alpha should be in range -pi<alpha<pi')
            alpha_range = np.ones(num_points)*i[1]
        if i[0] == 'x_1':
            if i[1] > 1:
                raise Exception('Error: x_1 should be < 1')
            x_1_range = np.ones(num_points)*i[1]
        if i[0] == 'x_2':
            if i[1] > 1 or i[1] < (1-x_1_range[0]):
                raise Exception('Error: x_2 should be in range 1-x_1 < x_2 < 1')
            x_2_range = np.ones(num_points)*i[1]
            x_3_range = 2-x_1_range-x_2_range
            
    # assume if x_2 not fixed, but x_1 is then just vary x_2 and x_3        
    if 'x_1' in fix and 'x_2' not in fix:
        x_1sub = 1-x_1_range
            
        x_2_range = []
        for i in x_1sub:
            x_2 = np.random.uniform(i,1,1)
            x_2_range.append(x_2[0])
        x_2_range = np.array(x_2_range)
        
        x_3_range = 2-x_1_range-x_2_range
        
            
    momenta = []
    soft = []
    for i in range(len(theta_range)):
        variables = [theta_range[i],alpha_range[i], x_1_range[i], x_2_range[i]]
        if is_soft(variables) == False:
            soft.append(False)
        else:
            soft.append(True)
        p_3, p_4, p_5 = out_particles_2_3(theta_range[i],alpha_range[i], x_1_range[i], x_2_range[i])
        momenta.append([p_1,p_2,p_3,p_4,p_5])
        
    close = []
    for i in tqdm(momenta):
        close.append(check_all(i))
            
    data = {'momenta':list(momenta), 'theta':list(theta_range), 'alpha':list(alpha_range), 'x1':list(x_1_range), 'x2':list(x_2_range), 'x3':list(x_3_range), 'too close':list(close), 'is soft':list(soft)}
    
    df_2_3 = pd.DataFrame(data)
    cut_2_3 = df_2_3[df_2_3['too close'] == False].reset_index()
    cut_2_3 = cut_2_3[cut_2_3['is soft'] == False].reset_index()
    
    cut_mom = cut_2_3['momenta']
    cut_theta = cut_2_3['theta']
    cut_alpha = cut_2_3['alpha']
    cut_x_1 = cut_2_3['x1']
    cut_x_2 = cut_2_3['x2']
    cut_x_3 = cut_2_3['x3']
    
    return cut_mom, cut_theta, cut_alpha, cut_x_1, cut_x_2, cut_x_3
    
    
    