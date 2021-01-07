import numpy as np

from n3jet.phase import (
    Rambo, check_all
)
from n3jet.phase.rambo import (
    dot,
    pair_check
)

def test__dot():
    p1 = [1.,2.,3.,4.]
    p2 = [5.,6.,7.,8.]

    assert dot(p1,p2) == 5.-(12.+21.+32.)

def test__pair_check():
    p1 = [5.,0.5,0.,-5.]
    p2 = [-4.,1.,1.,4.]

    close, distance = pair_check(p1,p2,delta=0.01,s_com=dot(p1,p2))
    
    assert close == False

def test__rambo_generate():

    delta_cut = 0.01
    
    rambo = Rambo(
        num_jets=3,
        num_points=100,
        w=500.,
        delta_cut=delta_cut,
        all_legs=False
    )

    cut_mom = rambo.generate()

    assert np.isclose(dot(cut_mom[0][0],cut_mom[0][0]),0.)
    assert np.isclose(dot(cut_mom[0][1],cut_mom[0][1]),0.)
    assert np.isclose(dot(cut_mom[0][2],cut_mom[0][2]),0.)
    assert np.isclose(dot(cut_mom[0][3],cut_mom[0][3]),0.)
    assert np.isclose(dot(cut_mom[0][4],cut_mom[0][4]),0.)

    for i in cut_mom:    
        too_close, distance = check_all(
            mom=i,
            delta=delta_cut,
            s_com=dot(cut_mom[0][0],cut_mom[0][1]),
            all_legs=False
        )

        assert too_close == False
        assert distance > delta_cut

def test__rambo_generate_all_legs():

    delta_cut = 0.01
    
    rambo = Rambo(
        num_jets=3,
        num_points=100,
        w=500.,
        delta_cut=delta_cut,
        all_legs=True
    )

    cut_mom = rambo.generate()

    assert np.isclose(dot(cut_mom[0][0],cut_mom[0][0]),0.)
    assert np.isclose(dot(cut_mom[0][1],cut_mom[0][1]),0.)
    assert np.isclose(dot(cut_mom[0][2],cut_mom[0][2]),0.)
    assert np.isclose(dot(cut_mom[0][3],cut_mom[0][3]),0.)
    assert np.isclose(dot(cut_mom[0][4],cut_mom[0][4]),0.)

    for i in cut_mom:    
        too_close, distance = check_all(
            mom=i,
            delta=delta_cut,
            s_com=dot(cut_mom[0][0],cut_mom[0][1]),
            all_legs=True
        )

        assert too_close == False
        assert distance > delta_cut

def test__rambo_generate_piecewise():

    delta_cut = 0.01
    delta_near = 0.02

    rambo = Rambo(
        num_jets=3,
        num_points=100,
        w=500.,
        delta_cut=delta_cut,
        all_legs=False
    )

    cut_mom, near_mom = rambo.generate_piecewise(delta_near=delta_near)

    assert np.isclose(dot(cut_mom[0][0],cut_mom[0][0]),0.)
    assert np.isclose(dot(cut_mom[0][1],cut_mom[0][1]),0.)
    assert np.isclose(dot(cut_mom[0][2],cut_mom[0][2]),0.)
    assert np.isclose(dot(cut_mom[0][3],cut_mom[0][3]),0.)
    assert np.isclose(dot(cut_mom[0][4],cut_mom[0][4]),0.)

    assert np.isclose(dot(near_mom[0][0],near_mom[0][0]),0.)
    assert np.isclose(dot(near_mom[0][1],near_mom[0][1]),0.)
    assert np.isclose(dot(near_mom[0][2],near_mom[0][2]),0.)
    assert np.isclose(dot(near_mom[0][3],near_mom[0][3]),0.)
    assert np.isclose(dot(near_mom[0][4],near_mom[0][4]),0.)

    for i in cut_mom:    
        too_close, distance = check_all(
            mom=i,
            delta=delta_cut,
            s_com=dot(cut_mom[0][0],cut_mom[0][1]),
            all_legs=False
        )

        assert too_close == False
        assert distance > delta_cut
        assert distance > delta_near

    for i in near_mom:    
        too_close, distance = check_all(
            mom=i,
            delta=delta_cut,
            s_com=dot(cut_mom[0][0],cut_mom[0][1]),
            all_legs=False
        )

        assert too_close == False
        assert distance > delta_cut
        assert distance < delta_near

def test__rambo_generate_piecewise_all_legs():

    delta_cut = 0.01
    delta_near = 0.02

    rambo = Rambo(
        num_jets=3,
        num_points=100,
        w=500.,
        delta_cut=delta_cut,
        all_legs=True
    )

    cut_mom, near_mom = rambo.generate_piecewise(delta_near=delta_near)

    assert np.isclose(dot(cut_mom[0][0],cut_mom[0][0]),0.)
    assert np.isclose(dot(cut_mom[0][1],cut_mom[0][1]),0.)
    assert np.isclose(dot(cut_mom[0][2],cut_mom[0][2]),0.)
    assert np.isclose(dot(cut_mom[0][3],cut_mom[0][3]),0.)
    assert np.isclose(dot(cut_mom[0][4],cut_mom[0][4]),0.)

    assert np.isclose(dot(near_mom[0][0],near_mom[0][0]),0.)
    assert np.isclose(dot(near_mom[0][1],near_mom[0][1]),0.)
    assert np.isclose(dot(near_mom[0][2],near_mom[0][2]),0.)
    assert np.isclose(dot(near_mom[0][3],near_mom[0][3]),0.)
    assert np.isclose(dot(near_mom[0][4],near_mom[0][4]),0.)

    for i in cut_mom:    
        too_close, distance = check_all(
            mom=i,
            delta=delta_cut,
            s_com=dot(cut_mom[0][0],cut_mom[0][1]),
            all_legs=True
        )

        assert too_close == False
        assert distance > delta_cut
        assert distance > delta_near

    for i in near_mom:    
        too_close, distance = check_all(
            mom=i,
            delta=delta_cut,
            s_com=dot(cut_mom[0][0],cut_mom[0][1]),
            all_legs=True
        )

        assert too_close == False
        assert distance > delta_cut
        assert distance < delta_near
