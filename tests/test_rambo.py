from n3jet.phase.rambo import dot

def test__dot():
    p1 = [1.,2.,3.,4.]
    p2 = [5.,6.,7.,8.]

    assert dot(p1,p2) == 5.-(12.+21.+32.)
