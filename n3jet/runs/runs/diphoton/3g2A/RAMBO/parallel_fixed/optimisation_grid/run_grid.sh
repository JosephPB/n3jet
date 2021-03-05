#!/bin/bash

shopt -s nullglob
for d in */
do
   $d/run_grid.sh
   wait
done
