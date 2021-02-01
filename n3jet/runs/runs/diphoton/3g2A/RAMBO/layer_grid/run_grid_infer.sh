#!/bin/bash

shopt -s nullglob
for f in *.yaml
do
    python /mt/home/jbullock/n3jet/n3jet/general/fks/general_precision_error_testing.py --yaml_file $f &
done
