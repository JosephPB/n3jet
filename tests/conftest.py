import numpy as np
import pytest

@pytest.fixture(name="dummy_data", scope="session")
def make_dummy_data():

    delta_cut = 0.01
    delta_near = 0.02

    cut_mom = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([ 159.22451821,   -6.63307095,  -84.13071446, -135.02026682]),
        np.array([ 120.32231578, -109.77040355,  -49.14656498,    3.54024433]),
        np.array([220.45316601, 116.4034745 , 133.27727944, 131.48002249])
    ]

    near_mom = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([216.40580479, -70.68745046, -56.43043907, 196.5969538 ]),
        np.array([ 246.15021728,   86.55738039,   85.56729242, -213.95323749]),
        np.array([ 37.44397793, -15.86992993, -29.13685335,  17.3562837 ])
    ]
    
    momenta = []
    momenta.append(cut_mom)
    momenta.append(near_mom)

    near_momenta = [near_mom]
    cut_momenta = [cut_mom]

    cut_label = 1.
    near_label = 2.
    
    labels = []
    labels.append(cut_label)
    labels.append(near_label)

    return momenta, cut_momenta, near_momenta, labels, cut_label, near_label, delta_cut, delta_near

@pytest.fixture(name="dummy_data_training", scope="session")
def make_dummy_data_training():

    delta_cut = 0.01
    delta_near = 0.02

    cut_mom_1 = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([ 159.22451821,   -6.63307095,  -84.13071446, -135.02026682]),
        np.array([ 120.32231578, -109.77040355,  -49.14656498,    3.54024433]),
        np.array([220.45316601, 116.4034745 , 133.27727944, 131.48002249])
    ]

    cut_mom_2 = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([ 169.69039593,  -92.35396574, -142.34479507,    1.88010654]),
        np.array([113.95042459, -99.81405895,  46.70228226, -28.99568468]),
        np.array([216.35917948, 192.16802469,  95.6425128 ,  27.11557814])
    ]

    
    near_mom_1 = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([216.40580479, -70.68745046, -56.43043907, 196.5969538 ]),
        np.array([ 246.15021728,   86.55738039,   85.56729242, -213.95323749]),
        np.array([ 37.44397793, -15.86992993, -29.13685335,  17.3562837 ])
    ]

    near_mom_2 = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([ 55.18060235,  -9.9068167 , -52.66540422,  13.1570915 ]),
        np.array([ 198.93004896,   23.75856402, -138.76107726,  140.54913181]),
        np.array([ 245.88934869,  -13.85174733,  191.42648148, -153.70622331])
    ]
    
    near_momenta = []
    cut_momenta = []
    momenta = []
    for i in range(5):
        near_momenta.append(near_mom_1)
        cut_momenta.append(cut_mom_1)
        momenta.append(near_mom_1)
        momenta.append(cut_mom_1)

    for i in range(5):
        near_momenta.append(near_mom_2)
        cut_momenta.append(cut_mom_2)
        momenta.append(near_mom_2)
        momenta.append(cut_mom_2)
        
    
    cut_label_1 = 1.
    cut_label_2 = 1.5
    near_label_1 = 2.
    near_label_2 = 2.5

    cut_labels = []
    near_labels = []
    labels = []
    for i in range(5):
        near_labels.append(near_label_1)
        cut_labels.append(cut_label_1)
        labels.append(near_label_1)
        labels.append(cut_label_1)

    for i in range(5):
        near_labels.append(near_label_2)
        cut_labels.append(cut_label_2)
        labels.append(near_label_2)
        labels.append(cut_label_2)

    return momenta, cut_momenta, near_momenta, labels, cut_labels, near_labels, delta_cut, delta_near

@pytest.fixture(name="dummy_data_all_legs", scope="session")
def make_dummy_data_all_legs():

    delta_cut = 0.01
    delta_near = 0.02

    cut_mom = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([ 159.22451821,   -6.63307095,  -84.13071446, -135.02026682]),
        np.array([ 120.32231578, -109.77040355,  -49.14656498,    3.54024433]),
        np.array([220.45316601, 116.4034745 , 133.27727944, 131.48002249])
    ]

    near_mom_all_legs =[
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([114.02383821,  29.90221969,  92.19870798, -60.05573399]),
        np.array([191.46252681, -15.42504935,  59.08235119, 181.46416396]),
        np.array([ 194.51363498,  -14.47717034, -151.28105917, -121.40842998])
    ]

    momenta_all_legs = []
    momenta_all_legs.append(cut_mom)
    momenta_all_legs.append(near_mom_all_legs)

    near_momenta_all_legs = [near_mom_all_legs]
    cut_momenta = [cut_mom]

    cut_label = 1.
    near_label = 2.
    
    labels = []
    labels.append(cut_label)
    labels.append(near_label)

    return momenta_all_legs, cut_momenta, near_momenta_all_legs, labels, cut_label, near_label, delta_cut, delta_near

@pytest.fixture(name="dummy_data_all_legs_training", scope="session")
def make_dummy_data_all_legs_training():

    delta_cut = 0.01
    delta_near = 0.02

    cut_mom_1 = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([ 159.22451821,   -6.63307095,  -84.13071446, -135.02026682]),
        np.array([ 120.32231578, -109.77040355,  -49.14656498,    3.54024433]),
        np.array([220.45316601, 116.4034745 , 133.27727944, 131.48002249])
    ]

    cut_mom_2 = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([ 169.69039593,  -92.35396574, -142.34479507,    1.88010654]),
        np.array([113.95042459, -99.81405895,  46.70228226, -28.99568468]),
        np.array([216.35917948, 192.16802469,  95.6425128 ,  27.11557814])
    ]

    near_mom_all_legs_1 =[
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([114.02383821,  29.90221969,  92.19870798, -60.05573399]),
        np.array([191.46252681, -15.42504935,  59.08235119, 181.46416396]),
        np.array([ 194.51363498,  -14.47717034, -151.28105917, -121.40842998])
    ]

    near_mom_all_legs_2 = [
        np.array([250.,   0.,   0., 250.]),
        np.array([ 250.,    0.,    0., -250.]),
        np.array([243.07408153,  12.42726463, 228.81032092,  81.095063  ]),
        np.array([ 226.72827016,    4.78438267, -219.05776658,  -58.27961112]),
        np.array([ 30.19764832, -17.2116473 ,  -9.75255435, -22.81545188])
    ]

    near_momenta_all_legs = []
    cut_momenta = []
    momenta_all_legs = []
    for i in range(5):
        near_momenta_all_legs.append(near_mom_all_legs_1)
        cut_momenta.append(cut_mom_1)
        momenta_all_legs.append(near_mom_all_legs_1)
        momenta_all_legs.append(cut_mom_1)

    for i in range(5):
        near_momenta_all_legs.append(near_mom_all_legs_2)
        cut_momenta.append(cut_mom_2)
        momenta_all_legs.append(near_mom_all_legs_2)
        momenta_all_legs.append(cut_mom_2)
    
    cut_label_1 = 1.
    cut_label_2 = 1.5
    near_label_1 = 2.
    near_label_2 = 2.5

    cut_labels = []
    near_labels = []
    labels = []
    for i in range(5):
        near_labels.append(near_label_1)
        cut_labels.append(cut_label_1)
        labels.append(near_label_1)
        labels.append(cut_label_1)

    for i in range(5):
        near_labels.append(near_label_2)
        cut_labels.append(cut_label_2)
        labels.append(near_label_2)
        labels.append(cut_label_2)

    
    return momenta_all_legs, cut_momenta, near_momenta_all_legs, labels, cut_labels, near_labels, delta_cut, delta_near
