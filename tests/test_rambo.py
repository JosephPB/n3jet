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

    rambo = Rambo(
        num_jets=3,
        num_points=10,
        w=500.,
        delta_cut=0.01,
        all_legs=True
    )

    cut_mom = rambo.generate()

    assert dot(cut_mom[0][0],cut_mom[0][0]) == 0.
    assert dot(cut_mom[0][1],cut_mom[0][1]) == 0.
    assert dot(cut_mom[0][2],cut_mom[0][2]) == 0.
    assert dot(cut_mom[0][3],cut_mom[0][3]) == 0.
    assert dot(cut_mom[0][4],cut_mom[0][4]) == 0.
