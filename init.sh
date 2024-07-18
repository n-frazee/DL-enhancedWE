#!/bin/bash

# Set up simulation environment
source env.sh

# Clean up from previous/ failed runs
rm -rf traj_segs seg_logs istates west.h5 ddwe-logs
mkdir   seg_logs traj_segs istates

# Set pointer to bstate and tstate
#BSTATE_ARGS="--bstate-file $WEST_SIM_ROOT/bstates/bstates.txt"
#TSTATE_ARGS="--tstate-file $WEST_SIM_ROOT/tstate.file"

#BSTATE_ARGS = ''

# Run w_init
#w_init \
#    $BSTATE_ARGS \
#    $TSTATE_ARGS \
#  --segs-per-state 36 \
#  --work-manager=threads "$@" > init.log

# Note all states from the "clusters" in bstates.txt are from RMSD > 10.2 Ang
# The tstate would be structures with RMSD < 1 Ang

w_init --bstate-file "ntl9_folding_synd/bstates.txt" \
        --tstate-file "tstate.file" \
	--segs-per-state 2 \
	--work-manager=threads "$@" > init.log
