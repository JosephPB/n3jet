import numpy as np
import yaml
import pytest

from n3jet.general import FKSModelRun
from n3jet.paths import configs_path

example_config = configs_path / "example_config.yaml"

def test__yaml_readin():

    example_config = configs_path / "example_config.yaml"

    print (example_config)

    fksmodel = FKSModelRun.from_yaml(yaml_file = example_config)
    
    with open(example_config) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)

    assert fksmodel.mom_file == y["training"]["mom_file"]
    assert fksmodel.nj_file == y["training"]["nj_file"]
    assert fksmodel.delta_cut == y["delta_cut"]
    assert fksmodel.delta_near == y["delta_near"]

def test__split_data(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    fksmodel = FKSModelRun.from_yaml(example_config)
    fksmodel.delta_cut = 0.0
    fksmodel.delta_near = 0.02

    cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split = fksmodel.split_data(momenta, labels)

    assert len(np.where(np.all(cut_momenta==cut_mom[0],axis=(1,2)))[0]) > 0
    assert len(np.where(cut_nj==cut_labs[0])[0]) > 0
