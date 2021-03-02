import numpy as np
import yaml
import pytest

from n3jet.general import FKSModelRun

example_config = "example_config.yaml"

def test__yaml_readin():
    
    with open(yaml_file) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)

    fksmodel = FKSModelRun.from_yaml(example_config)

    assert fksmodel.mom_file == y["training"]["mom_file"]
    assert fksmodel.nj_file == y["training"]["nj_file"]
    assert fksmodel.delta_cut == y["delta_cut"]
    assert fksmodel.delta_near == y["delta_near"]

def test__split_data(dummy_data_all_legs_training):

    momenta, cut_mom, near_mom, labels, cut_labs, near_labs, delta_cut, delta_near = dummy_data_all_legs_training

    fksmodel = FKSModelRun.from_yaml(example_config)
