import numpy as np
import yaml
import pytest

from n3jet.general import SingleModelRun

example_config = "./configs/example_config.yaml"

def test__yaml_readin():

    singlemodel = SingleModelRun.from_yaml(yaml_file = example_config)
    
    with open(example_config) as f:
        y = yaml.load(f, Loader=yaml.FullLoader)

    assert singlemodel.mom_file == y["training"]["mom_file"]
    assert singlemodel.nj_file == y["training"]["nj_file"]
    assert singlemodel.delta_cut == y["delta_cut"]
    assert singlemodel.delta_near == y["delta_near"]

def test__recut_data()
