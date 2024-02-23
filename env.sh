#!/bin/bash

# Set up environment for dynamics
source $AMBERHOME/amber.sh

# Set WESTPA-related variables
export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename $WEST_SIM_ROOT)

# Set runtime commands
export PMEMD=$(which pmemd)
export SANDER=$(which sander)
export CPPTRAJ=$(which cpptraj)
