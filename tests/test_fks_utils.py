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

def test__train_near_network(dummy_data_training):

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
