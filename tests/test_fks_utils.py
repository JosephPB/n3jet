import numpy as np

from n3jet.utils import FKSPartition

from n3jet.utils.fks_utils import(
    train_near_networks,
    train_near_networks_general,
    train_cut_network,
    train_cut_network_general,
    infer_on_near_splits,
    infer_on_near_splits_separate,
    infer_on_cut
)

from n3jet.models import Model

def test__train_near_networks(dummy_data_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_training 
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    

    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks(
        pairs = pairs,
        near_momenta = near_momenta,
        NJ_split = labs_split,
        order = 'LO',
        n_gluon = 1,
        delta_near = delta_near,
        points = len(near_momenta)*2,
        model_dir = '',
        epochs = 1
    )


def test__train_near_networks_general(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    

    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks_general(
        input_size = (nlegs+2)*4,
        pairs = pairs,
        near_momenta = near_momenta,
        NJ_split = labs_split,
        delta_near = delta_near,
        model_dir = '',
        all_jets = True,
        all_legs = True,
        model_dataset = False,
        epochs = 1,
    )

def test__train_near_networks_general_dataset(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    

    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks_general(
        input_size = (nlegs+2)*4,
        pairs = pairs,
        near_momenta = near_momenta,
        NJ_split = labs_split,
        delta_near = delta_near,
        model_dir = '',
        all_jets = True,
        all_legs = True,
        model_dataset = True,
        epochs = 1,
    )

def test__train_near_networks_general_high_precision(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    

    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks_general(
        input_size = (nlegs+2)*4,
        pairs = pairs,
        near_momenta = near_momenta,
        NJ_split = labs_split,
        delta_near = delta_near,
        model_dir = '',
        all_jets = True,
        all_legs = True,
        model_dataset = False,
        epochs = 1,
        high_precision = True
    )

def test__train_cut_network(dummy_data_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_training 
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = train_cut_network(
        cut_momenta = cut_momenta,
        NJ_cut = cut_labels,
        order = 'LO',
        n_gluon = 1,
        delta_cut = delta_cut,
        points = len(cut_momenta)*2,
        model_dir = '',
        epochs = 1
    )

def test__train_cut_network_general(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = train_cut_network_general(
        input_size = (nlegs+2)*4,
        cut_momenta = cut_momenta,
        NJ_cut = cut_labels,
        delta_cut = delta_cut,
        model_dir = '',
        all_jets = True,
        all_legs = True,
        model_dataset = False,
        epochs = 1
    )

def test__train_cut_network_general_dataset(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = train_cut_network_general(
        input_size = (nlegs+2)*4,
        cut_momenta = cut_momenta,
        NJ_cut = cut_labels,
        delta_cut = delta_cut,
        model_dir = '',
        all_jets = True,
        all_legs = True,
        model_dataset = True,
        epochs = 1
    )

def test__train_cut_network_general_high_precision(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = train_cut_network_general(
        input_size = (nlegs+2)*4,
        cut_momenta = cut_momenta,
        NJ_cut = cut_labels,
        delta_cut = delta_cut,
        model_dir = '',
        all_jets = True,
        all_legs = True,
        epochs = 1,
        high_precision = True
    )

def test__infer_on_near_splits(dummy_data_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_training 
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    

    n_gluon = 1
    
    NN = Model((n_gluon+2-1)*4,near_momenta,labs_split[0],all_jets=False,all_legs=False)
    _,_,_,_,_,_,_,_ = NN.process_training_data()
    
    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks(
        pairs = pairs,
        near_momenta = near_momenta,
        NJ_split = labs_split,
        order = 'LO',
        n_gluon = 1,
        delta_near = delta_near,
        points = len(near_momenta)*2,
        model_dir = '',
        epochs = 1
    )
    
    y_pred_nears = infer_on_near_splits(
        NN = NN,
        moms = near_momenta,
        models = model_near,
        x_mean_near = x_mean_near,
        x_std_near = x_std_near,
        y_mean_near = y_mean_near,
        y_std_near = y_std_near
    )

def test__infer_on_near_splits_separate(dummy_data_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_training 
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    

    n_gluon = 1
    
    NN = Model((n_gluon+2-1)*4,near_momenta,labs_split[0],all_jets=False,all_legs=False)
    _,_,_,_,_,_,_,_ = NN.process_training_data()
    
    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks(
        pairs = pairs,
        near_momenta = near_momenta,
        NJ_split = labs_split,
        order = 'LO',
        n_gluon = 1,
        delta_near = delta_near,
        points = len(near_momenta)*2,
        model_dir = '',
        epochs = 1
    )
    
    y_preds_nears, y_pred_nears = infer_on_near_splits_separate(
        NN = NN,
        moms = near_momenta,
        models = model_near,
        x_mean_near = x_mean_near,
        x_std_near = x_std_near,
        y_mean_near = y_mean_near,
        y_std_near = y_std_near
    )

def test__infer_on_cut(dummy_data_training):
    
    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_training 
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)

    n_gluon = 1

    NN = Model((n_gluon+2-1)*4,cut_momenta,cut_labels,all_jets=False,all_legs=False)
    _,_,_,_,_,_,_,_ = NN.process_training_data()

    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = train_cut_network(
        cut_momenta = cut_momenta,
        NJ_cut = cut_labels,
        order = 'LO',
        n_gluon = 1,
        delta_cut = delta_cut,
        points = len(cut_momenta)*2,
        model_dir = '',
        epochs = 1
    )

    y_pred_cuts = infer_on_cut(
        NN = NN,
        moms = cut_momenta,
        model = model_cut,
        x_mean_cut = x_mean_cut,
        x_std_cut = x_std_cut,
        y_mean_cut = y_mean_cut,
        y_std_cut = y_std_cut
    )
