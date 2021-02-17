#!/bin/bash

shopt -s nullglob
for f in *.yaml
do
    python /mt/home/jbullock/n3jet/n3jet/general/fks/general_init_model_rerun.py --yaml_file $f &
done
