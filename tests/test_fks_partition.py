import numpy as np

from n3jet.utils import FKSPartition


def check_list(l1, l2):
    for idx, i in enumerate(l1):
        if l2[idx] != i:
            return False
    return True

def test__cut_near_split(dummy_data):

    momenta, cut_mom, near_mom, labels, cut_label, near_label, delta_cut, delta_near = dummy_data
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    assert check_list(cut_momenta[0][2],cut_mom[0][2])
    assert check_list(near_momenta[0][2], near_mom[0][2])
    assert cut_labels[0] == cut_label
    assert near_labels[0] == near_label

def test__cut_near_split_all_legs(dummy_data_all_legs):

    momenta_all_legs, cut_mom, near_mom_all_legs, labels, cut_label, near_label, delta_cut, delta_near = dummy_data_all_legs
    
    fks = FKSPartition(
        momenta = momenta_all_legs,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    assert check_list(cut_momenta[0][2],cut_mom[0][2])
    assert check_list(near_momenta[0][2], near_mom_all_legs[0][2])
    assert cut_labels[0] == cut_label
    assert near_labels[0] == near_label

def test__weighting(dummy_data):

    momenta, cut_mom, near_mom, labels, cut_label, near_label, delta_cut, delta_near = dummy_data
    
    fks = FKSPartition(
           momenta = momenta,
           labels = labels,
           all_legs = False
       )
    
    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()

    assert len(pairs) == 3
    assert len(labs_split) == 3
    assert np.isclose(np.sum(labs_split),near_label)

def test__weighting_all_legs(dummy_data_all_legs):

    momenta_all_legs, cut_mom, near_mom_all_legs, labels, cut_label, near_label, delta_cut, delta_near = dummy_data_all_legs
    
    fks = FKSPartition(
           momenta = momenta_all_legs,
           labels = labels,
           all_legs = True
       )
    
    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()

    assert len(pairs) == 9
    assert len(labs_split) == 9
    assert np.isclose(np.sum(labs_split),near_label)
