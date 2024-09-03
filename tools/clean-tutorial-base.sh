#!/usr/bin/env sh
set -e -u

# shellcheck disable=SC1091
. ../tools/cleaning-tools.sh

clean_tutorial .
clean_precice_logs .
rm -fv **/*.log
rm -fv **/*.vtu
rm -fv **/*.vtk

# Cluster related files to clean
rm -fv *.err
rm -fv *.out
