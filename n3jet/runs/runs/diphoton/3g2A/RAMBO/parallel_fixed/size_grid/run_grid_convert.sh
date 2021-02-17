#!/bin/bash

shopt -s nullglob
for f in *.yaml
do
    python /mt/home/jbullock/n3jet/n3jet/c++_calls/conversion/modeldump_multiple_fks.py --yaml_file $f -p 9 -ob /mt/home/jbullock/n3jet/n3jet/c++_calls/models/diphoton/3g2A/RAMBO/parallel_fixed/size_grid/ &
done
