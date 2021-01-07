from n3jet.phase import Rambo
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
        num_points=10,
        w=500.,
        delta_cut=delta_cut,
        all_legs=True
    )

    cut_mom = rambo.generate()

    assert dot(cut_mom[0][0],cut_mom[0][0]) == 0.
    assert dot(cut_mom[0][1],cut_mom[0][1]) == 0.
    assert dot(cut_mom[0][2],cut_mom[0][2]) == 0.
    assert dot(cut_mom[0][3],cut_mom[0][3]) == 0.
    assert dot(cut_mom[0][4],cut_mom[0][4]) == 0.

    too_close, distance = check_all(cut_mom, delta_cut, dot(cut_mom[0][0], cut_mom[0][1]))

    assert too_close == False
    assert distance > delta_cut

def test__rambo_generate_piecewise():

    delta_cut = 0.01
    delta_near = 0.02

    rambo = Rambo(
        num_jets=3,
        num_points=10,
        w=500.,
        delta_cut=delta_cut,
        all_legs=True
    )

    cut_mom, near_mom = rambo.generate(delta_near=delta_near)

    assert dot(cut_mom[0][0],cut_mom[0][0]) == 0.
    assert dot(cut_mom[0][1],cut_mom[0][1]) == 0.
    assert dot(cut_mom[0][2],cut_mom[0][2]) == 0.
    assert dot(cut_mom[0][3],cut_mom[0][3]) == 0.
    assert dot(cut_mom[0][4],cut_mom[0][4]) == 0.

    assert dot(near_mom[0][0],near_mom[0][0]) == 0.
    assert dot(near_mom[0][1],near_mom[0][1]) == 0.
    assert dot(near_mom[0][2],near_mom[0][2]) == 0.
    assert dot(near_mom[0][3],near_mom[0][3]) == 0.
    assert dot(near_mom[0][4],near_mom[0][4]) == 0.

    too_close, distance = check_all(cut_mom[0], delta_cut, dot(cut_mom[0][0], cut_mom[0][1]))

    assert too_close == False
    assert distance > delta_near
    assert distance > delta_cut

    too_close, distance = check_all(near_mom[0], delta_cut, dot(cut_mom[0][0], cut_mom[0][1]))

    assert too_close == False
    assert distance < delta_near
    assert distance > delta_cut
