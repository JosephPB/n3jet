import numpy as np

from n3jet.utils import FKSPartition

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

near_mom_all_legs =[
    np.array([250.,   0.,   0., 250.]),
    np.array([ 250.,    0.,    0., -250.]),
    np.array([114.02383821,  29.90221969,  92.19870798, -60.05573399]),
    np.array([191.46252681, -15.42504935,  59.08235119, 181.46416396]),
    np.array([ 194.51363498,  -14.47717034, -151.28105917, -121.40842998])
]

cut_label = 1.
near_label = 2.


momenta = []
momenta.append(cut_mom)
momenta.append(near_mom)

momenta_all_legs = []
momenta_all_legs.append(cut_mom)
momenta_all_legs.append(near_mom_all_legs)

labels = []
labels.append(cut_label)
labels.append(near_label)

def test__cut_near_split():

    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    # assert cut_momenta == cut_mom
    # assert near_momenta == near_mom
    # assert cut_labels == cut_label
    # assert near_labels == near_label

def test__cut_near_split_all_legs():

    fks = FKSPartition(
        momenta = momenta_all_legs,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    # assert cut_momenta == cut_mom
    # assert near_momenta == near_mom
    # assert cut_labels == cut_label
    # assert near_labels == near_label

def test__weighting():
    
    fks = FKSPartition(
           momenta = momenta,
           labels = labels,
           all_legs = False
       )
    
    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()

    assert len(pairs) == 3
    # assert labs_split[0] == ?

def test__weighting_all_legs():
    
    fks = FKSPartition(
           momenta = momenta_all_legs,
           labels = labels,
           all_legs = True
       )
    
    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()

    assert len(pairs) == 9
    # assert labs_split[0] == ?
