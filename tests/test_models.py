import numpy as np
import pytest

from n3jet.utils import FKSPartition
from n3jet.models import Model

@pytest.fixture(name="model")
def create_model(dummy_data_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_training

    nlegs = len(momenta[0])-2
    n_gluon = 1
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = False
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()


    model = Model(
        input_size = (n_gluon+2-1)*4,
        momenta = cut_momenta,
        labels = cut_labels,
        all_jets = False,
        all_legs = False,
    )

    return model

@pytest.fixture(name="model_all_legs")
def create_model_all_legs(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    
    
    model = Model(
        input_size = (nlegs+2)*4,
        momenta = cut_momenta,
        labels = cut_labels,
        all_jets = False,
        all_legs = True,
        model_dataset = False
    )

    return model

@pytest.fixture(name="model_all_legs_dataset")
def create_model_all_legs_dataset(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    nlegs = len(momenta[0])-2
    
    fks = FKSPartition(
        momenta = momenta,
        labels = labels,
        all_legs = True
    )

    cut_momenta, near_momenta, cut_labels, near_labels = fks.cut_near_split(delta_cut, delta_near)
    pairs, labs_split = fks.weighting()    
    
    model = Model(
        input_size = (nlegs+2)*4,
        momenta = cut_momenta,
        labels = cut_labels,
        all_jets = False,
        all_legs = True,
        model_dataset = True
    )

    return model

def test__process_training_data(model, model_all_legs, model_all_legs_dataset):

    X_train, X_test, y_train, y_test, x_mean, x_std, y_mean, y_std = model.process_training_data()

    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    
    X_train, X_test, y_train, y_test, x_mean, x_std, y_mean, y_std = model_all_legs.process_training_data()

    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    
    X_train, X_test, y_train, y_test, x_mean, x_std, y_mean, y_std = model_all_legs_dataset.process_training_data()

    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    
def test__fit(model, model_all_legs, model_all_legs_dataset):

    model, x_mean, x_std, y_mean, y_std = model.fit(epochs=2)
    
    model, x_mean, x_std, y_mean, y_std = model_all_legs.fit(layers=[10,20,30,40], epochs=2)
    
    model, x_mean, x_std, y_mean, y_std = model_all_legs_dataset.fit(epochs=2, lr = 0.01)

def test__process_testing_data(
        model,
        model_all_legs,
        model_all_legs_dataset,
        dummy_data_training,
        dummy_data_all_legs_training
):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_training
    X_train, X_test, y_train, y_test, x_mean, x_std, y_mean, y_std = model.process_training_data()
    x_standard = model.process_testing_data(moms = cut_mom)
    x_standard, y_standard = model.process_testing_data(moms = cut_mom, labs = cut_labs)

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training
    X_train, X_test, y_train, y_test, x_mean, x_std, y_mean, y_std = model_all_legs.process_training_data()
    x_standard = model_all_legs.process_testing_data(moms = cut_mom)
    x_standard, y_standard = model_all_legs.process_testing_data(moms = cut_mom, labs = cut_labs)

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training
    X_train, X_test, y_train, y_test, x_mean, x_std, y_mean, y_std = model_all_legs_dataset.process_training_data()
    x_standard = model_all_legs_dataset.process_testing_data(moms = cut_mom)
    x_standard, y_standard = model_all_legs_dataset.process_testing_data(moms = cut_mom, labs = cut_labs)
