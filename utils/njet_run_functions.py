from __future__ import print_function
import os
import sys
import re
import getopt
import itertools
import time
from operator import add, sub
from tqdm import tqdm

from math import pi, sqrt

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

#import njet
import imp
NJET_DIR = '/mt/home/jbullock/njet/njet-develop/'
# NJET_LIB = NJET_DIR + /.libs/
NJET_LIB = '/mt/home/jbullock/local/lib/'
sys.path.append(NJET_DIR)
sys.path.append(NJET_DIR + '/examples/')

njet = imp.load_source('njet', os.path.join(os.path.dirname(__file__), NJET_DIR + '/blha/njet.py'))

OLP = njet.OLP

import testdata #folder containing testdata - TODO: No not require this data and should be able to custom input

DEBUG = False
SLCTEST = None
CCTEST = None
NPOINTS = 10000000
VIEW = 'NJ'
if sys.platform.startswith('linux'):
    LIBNJET = os.path.join(os.path.dirname(__file__), NJET_LIB + '/libnjet2.so')
elif sys.platform.startswith('darwin'):
    LIBNJET = os.path.join(os.path.dirname(__file__), NJET_DIR + '/libnjet2.dylib')
else:
    print ("Warning: unknown system '%s'. Library will probably fail to load." % sys.platform)
    LIBNJET = os.path.join(os.path.dirname(__file__), NJET_DIR + '/.libs/libnjet2.dll')

factorial = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]  # n!


# Create order file form scratch -> contract file generated automatically
ORDER_TPL = """
# fixed
CorrectionType          QCD
BLHA1TwoCouplings       yes

# changable
#Extra NJetMultiPrec 2
#Extra Precision 1e-20
Extra NJetPrintStats yes
Extra NJetVectorClass yes
# test-specific
%s

# process list
"""

def njet_init(order):
    'Initiates njet'
    if os.path.exists(LIBNJET):
        status = OLP.OLP_Start(order, libnjet=LIBNJET)
    else:
        print ("Warning: '%s' not found, trying system default." % LIBNJET)
        status = OLP.OLP_Start(order)
    if DEBUG:
        if status:
            print (OLP.contract)
        else:
            print (order)
    return status

def relerr(a, b):
    'Calculated relative error'
    if abs(a+b) != 0.:
        return abs(2.*(a-b)/(a+b))
    else:
        return abs(a-b)


def out_vals(pref, norm1, val1, error, VIEW = VIEW):
    'Returns values'
    
    if VIEW == 'MC':
        vals = (pref, val1, error)
    else:
        vals = (pref, val1/norm1, error) # VIEW is set by default to 'NJ'
    return vals
    
def pretty_print_results(p, j, params, rval, mode=None, VIEW = VIEW):
    'Set up calculation of results'
    if mode is None:
        mode = params.get('mode', 'PLAIN')
    alphas = params.get('as', 1.)
    ao2pi = 4.*pi*alphas/(8.*pi*pi)

    ep2, ep1, ep0, born = rval[:4]
    ep2_err, ep1_err, ep0_err = rval[4:]

    norm = ao2pi*born ## assuming ML implementation

    if mode == 'NJ1':
        norm = born

    vals0 = out_vals("A0", 1, born,'NO ERROR')
    vals1_2 = out_vals("A1_2", norm, ep2, ep2_err, VIEW=VIEW)
    vals1_1 = out_vals("A1_1", norm, ep1, ep1_err, VIEW=VIEW)
    vals1_0 = out_vals("A1_0", norm, ep0, ep0_err, VIEW=VIEW)
    
    # NOTE: Now we output all accuracies without requiring DEBUG
    
    return vals0, vals1_2, vals1_1, vals1_0

def pretty_print_time(times):
    diffs = [x-y for (x,y) in zip(times[1:], times[:-1])]
    diffs2 = [x*x for x in diffs]
    n = len(diffs)
    avg = sum(diffs)/n
    if n > 1:
        sig = sqrt((sum(diffs2) - n*avg*avg)/(n - 1))
        print ("Time: %.4f (%.2f%%) s" % (avg, 100*sig/avg))
    else:
        print ("Time: %.4f s" % (avg))
        
def get_perm_order(dsn, dslimit):
    'Get permutation order for limited permutations'
    p = range(1,factorial[dsn]+1)
    if len(p) != dslimit:
        tmp = list(itertools.permutations(range(dsn)))
        p = [tuple(reversed(j))>j and i or i+len(p) for i,j in zip(p, tmp)]
    return p

def get_swap2(lst, s0, s1):
    'Get copy of a lst with swapped elements s0 and s1'
    rval = list(lst)
    rval[s0], rval[s1] = rval[s1], rval[s0]
    return rval

def get_dsmoms(mom, ds0, ds1, perm_order, sym2):
    'Get a list of permutations of mom, from ds0 to ds1 + symmetrisation of sym2'
    symfac = 1
    dsmoms = (mom[:ds0]+list(x)+mom[ds1:] for x in itertools.permutations(sorted(mom[ds0:ds1])))
    dsmoms = [m for o,m in sorted(zip(perm_order, dsmoms))]
    if sym2:
        s0, s1 = sym2
        dsmoms.extend([get_swap2(x, s0-1, s1-1) for x in dsmoms])
        symfac = 2
    return dsmoms, symfac

def get_name(j, dsn, dslimit, sym2):
    'Get a pretty name for a point'
    s = "... point %d ..." % j
    if dsn > 1:
        s += " %d! (%d permutations)" % (dsn, dslimit)
    if sym2:
        s += " x%d" % len(sym2)
    return s

def get_vals(moms, mcn, **kwargs):
    'Call OLP to get values, averages over permutations'
    rvals = []
    avg = 0
    for m in moms:
        avg += 1
        tmpval = OLP.OLP_EvalSubProcess(mcn, m, **kwargs)
        rvals.append(tmpval)
    rval = map(sum, zip(*rvals))
    return [x/avg for x in rval]

def mom_dot(p1,p2):
    'return dot product of momenta in +--- Minkowski metric'
    p_0 = p1[0]*p2[0]
    p_1 = p1[1]*p2[1]
    p_2 = p1[2]*p2[2]
    p_3 = p1[3]*p2[3]
    return p_0**2 - p_1**2 - p_2**2 - p_3**2

def run_generic_test(mom, params, data, new_mur = None, VIEW = VIEW):
    '''
    :param mom: list of momenta
    :param params: test_data[1]
    :param data: test_data[2]
    :param new_mur: new renormalisation scale
    :param VIEW: 'NJ' for k-factor and 'MC' otherwise?
    '''
    
    alphas = params.get('as', 1.)
    alpha = params.get('ae', 1.)
    mur = params.get('mur', 1.)
    mode = params.get('mode', 'PLAIN')
    test_vals = []
    for p in data:    
        mcn = p['mcn'] # get process number

        dsn = p.get('dsn', (1,1)) #desymmetrisation tuple that may be provided in param file
        ds0, ds1 = dsn[0]-1, dsn[1]
        dsn = ds1 - ds0
        dslimit = p.get('dslimit', factorial[dsn])
        perm_order = get_perm_order(dsn, dslimit) #gets permutation for limited permutations - check this

        sym2 = p.get('sym2', None)

        if 'name' in p:
            name = p['name']
        else:
            name = repr(p['inc'] + p['out'])

        npoints = min(NPOINTS, len(mom))
        print ("-------- channel %s -------- (%d points)" % (name, npoints))
        times = [time.time()]
        point_vals = []
        for j in tqdm(range(npoints)):
            mj = mom[j]
            dsmoms, symfac = get_dsmoms(mom[j], ds0, ds1, perm_order, sym2)
            allmoms = dsmoms[:(dslimit*symfac)] # allmoms has a shape of (1,4,4)
            
            #calculating Mandelstam variables
            if new_mur != None:
                p1, p2, p3, p4 = allmoms[0]
                s = mom_dot(list(map(add,p1,p2)),list(map(add,p1,p2)))
                t = mom_dot(list(map(sub,p1,p3)),list(map(sub,p1,p3)))
                u = mom_dot(list(map(sub,p1,p4)),list(map(sub,p1,p4)))
            
            if not SLCTEST or SLCTEST == 'both' or not p.get('has_lc', None):
                if new_mur == 's':
                    rval = get_vals(allmoms, mcn, alphas=alphas, alpha=alpha, mur=s)
                elif new_mur == 't':
                    rval = get_vals(allmoms, mcn, alphas=alphas, alpha=alpha, mur=t)
                elif new_mur == 'u':
                    rval = get_vals(allmoms, mcn, alphas=alphas, alpha=alpha, mur=u)
                else:
                    rval = get_vals(allmoms, mcn, alphas=alphas, alpha=alpha, mur=mur)
                vals0, vals1_2, vals1_1, vals1_0 = pretty_print_results(p, j, params, rval, VIEW = VIEW)
                point_vals.append([vals0, vals1_2, vals1_1, vals1_0])
            
            ## check meaning of canonical list for a channel in has_chan_lc
            ## check meaning of SLCTEST
            if SLCTEST and p.get('has_lc', None):
                print ("+++ LC+SLC +++")
                rval_lc = get_vals(allmoms, mcn+1, alphas=alphas, alpha=alpha, mur=mur)
                rval_slc = get_vals(allmoms, mcn+2, alphas=alphas, alpha=alpha, mur=mur)
                rval_sum = map(sum, zip(rval_lc, rval_slc)[:4])
                rval_sum[3] = rval_sum[3]/2
                vals0, vals1_2, vals1_1, vals1_0 = pretty_print_results(p, j, params, rval_sum)
                point_vals.append([vals0, vals1_2, vals1_1, vals1_0])
                if DEBUG:
                    print ("*** LC/(LC+SLC) ***")
                    rval_lc_r = [x/y if y!=0 else 1 for x,y in zip(rval_lc, rval_sum)]
                    vals0, vals1_2, vals1_1, vals1_0 = pretty_print_results(p, j, params, rval_lc_r, onlyus=True, mode='NJ1')
                    point_vals.append([vals0, vals1_2, vals1_1, vals1_0])
                    print ("*** SLC/(LC+SLC) ***")
                    rval_slc_r = [x/y if y!=0 else 1 for x,y in zip(rval_slc, rval_sum)]
                    vals0, vals1_2, vals1_1, vals1_0 = pretty_print_results(p, j, params, rval_slc_r, onlyus=True, mode='NJ1')
                    point_vals.append([vals0, vals1_2, vals1_1, vals1_0])
                    if SLCTEST == 'both':
                        rval_sum_r = [x/y if y!=0 else 1 for x,y in zip(rval_sum, rval)]
                        if all(abs(x-1) < 1e-13 for x in rval_sum_r):
                            s = 'OK'
                        else:
                            s = 'FAIL'
                        print ("*** (LC+SLC)/FULL *** %s" % s)
                        vals0, vals1_2, vals1_1, vals1_0 = pretty_print_results(p, j, params, rval_sum_r, onlyus=True, mode='NJ1')
                        point_vals.append([vals0, vals1_2, vals1_1, vals1_0])
            times.append(time.time())
        pretty_print_time(times)
        test_vals.append(point_vals)
    return test_vals


## NOTE: have not implemented run_sc_test yet
## TODO: If necessary implement these tests here

def nis(i, j):
    return i+j*(j-1)/2 if i<=j else j+i*(i-1)/2

def run_cc_test(mom, params, data):
    alphas = params.get('as', 1.)
    alpha = params.get('ae', 1.)
    mur = params.get('mur', 1.)
    mode = params.get('mode', 'PLAIN')
    legs = len(mom[0])
    for p in data:
        mcn = p['mcn']
        print (mcn)
        if 'name' in p:
            name = p['name']
        else:
            name = repr(p['inc'] + p['out'])
        npoints = min(NPOINTS, len(mom))
        print ("-------- channel %s -------- (%d points)" % (name, npoints))
        treevals = []
        for j in tqdm(range(npoints)):
            #print "... point %d ..." % j
            #ccvals = OLP.OLP_EvalSubProcess(mcn+1, mom[j], alphas=alphas, alpha=alpha, mur=mur, retlen=legs*(legs-1)/2)
            treeval = OLP.OLP_EvalSubProcess(mcn+2, mom[j], alphas=alphas, alpha=alpha, mur=mur)
            born = treeval[0]
            treevals.append(treeval)
            #print ('ccvals = {}'.format(ccvals))
            #print ('treeval = {}'.format(treeval))
            #print ('born = {}'.format(born))
            #if relerr(p['born'][j], born) > 1e-10:
            #    msg = 'FAIL'
            #else:
            #    msg = 'OK'
            if born == 0:
                print ("ERROR born = 0")
                born = 1
            #print p['born'][j]/born, msg
            #for i in range(legs):
            #    xsum = 0
            #    for j in range(legs):
            #        if i == j:
            #            sys.stdout.write(" %10.3e" % 0.)
            #        else:
            #            x = ccvals[nis(i,j)]/born
            #            sys.stdout.write(" %10.3e" % x)
            #            xsum += x
            #    sys.stdout.write(" | %17.10e\n" % (xsum))
    return treevals

def chan_has_lc(p):
    channel = njet.Channel([njet.Process.cross_flavour(i) for i in p['inc']] + p['out'])
    chanmatches = njet.Process.canonical.get(channel.canon_list, None)
    if chanmatches:
        return chanmatches[0].has_lc
    return False

def add_to_order(mcn, order, test):
    'Adding options to order file'
    
    new = ["\n"]
    params = test['params']
    if CCTEST:
        params['type'] = CCTEST
    ptype = params.get('type', 'CC')
    new.append("AlphasPower %d" % params.get('aspow', 0))
    new.append("AlphaPower  %d" % params.get('aepow', 0))
    if ptype == 'DS':
        new.append("AmplitudeType LoopDS")
    else:
        new.append("AmplitudeType Loop")
    new.append(params.get('order', ''))
    for p in test['data']:
        mcn += 1
        p['mcn'] = mcn
        procline = "%s -> %s" % (' '.join(map(str,p['inc'])),' '.join(map(str,p['out'])))
        new.append(procline)
        if ptype == 'NORMAL':
            p['has_lc'] = chan_has_lc(p) # if 'NORMAL' p['has_lc'] overwritten here
            if p['has_lc']:
                new.append("Process %d AmplitudeType LoopLC" % (mcn+1))
                new.append(procline)
                new.append("Process %d AmplitudeType LoopSLC" % (mcn+2))
                new.append(procline)
                mcn += 2
        elif ptype == 'CC':
            new.append("Process %d AmplitudeType ccTree" % (mcn+1))
            new.append(procline)
            new.append("Process %d AmplitudeType Tree" % (mcn+2))
            new.append(procline)
            mcn += 2
        elif ptype == 'SC':
            new.append("Process %d AmplitudeType scTree" % (mcn+1))
            new.append(procline)
            new.append("Process %d AmplitudeType Tree" % (mcn+2))
            new.append(procline)
            mcn += 2

    order += '\n'.join(new)
    return (mcn, order, ptype)

def run_batch(curorder, curtests):
    if DEBUG:
        curorder = "NJetReturnAccuracy 2\n" + curorder
    order = ORDER_TPL % curorder
    mcn = 0
    seen = []
    for t in curtests:
        test = t['test']
        if test not in seen:
            mcn, order, ptype = add_to_order(mcn, order, test)
            seen.append(test)
    if not njet_init(order):
        print ("Skipping batch due to errors")
        return

    test_data = []
    for t in curtests:
        proc = t['proc']
        mom = t['mod'].momenta
        test = t['test']
        params = test['params']
        if proc:
            data = [d for d in test['data'] if d['name'] == proc]
        else:
            data = test['data']
        if not data:
            print ("Warning: can't find %s" % proc)
            continue
        
        test_data.append([mom, params, data])

    ## NOTE: Eliminated running of generis tests and leave this to user    
        
    return test_data, ptype, order

def order_global(mod):
    'Add global order file elements to be added to tmp order file'
    order = []
    order.append('IRregularisation %s' % mod.scheme)
    if mod.renormalized:
        order.append('Extra NJetRenormalize yes')
    else:
        order.append('Extra NJetRenormalize no')
    #order.append('Extra SetParameter qcd(%d)' % mod.Nf)
    order.append(mod.extraorder)
    order = '\n'.join(order).rstrip(' \n')
    order = re.sub(r'\n\n+', r'\n', order)
    order = re.sub(r'\s\s+', r' ', order)
    return order

def run_tests(mods, tests):
    'Create order and test files in memory'
    cmporder_tmp = [order_global(m) for m in mods]
    cmporder = lambda x,y: cmp(cmporder_tmp.index(x[0]), cmporder_tmp.index(y[0]))
    sortmods = sorted([(order_global(m), m) for m in mods], cmp=cmporder)
    curorder = ''
    curtests = []
    for order,m in sortmods:
        if order != curorder:
            if curorder:
                run_batch(curorder, curtests)
            curorder = order
            curtests = [t for t in tests if t['mod'] == m]
        else:
            curtests.extend([t for t in tests if t['mod'] == m])
    return curorder, curtests

def action_run(param_file):
    '''
    Rewritten action_run taking param file as input
    example input: 'NJ_2J'
    '''

    modname, testname, proc = (param_file.split(':') + ['', ''])[:3]
    m = getattr(testdata, modname, None)
    mods = []
    tests = []
    
    mods.append(m)
    testname = m.groups
    for t in testname:
        tests.append({'mod' : m,
                      'test' : getattr(m, t),
                      'testname' : t,
                      'proc' : proc})
    
    return mods, tests



