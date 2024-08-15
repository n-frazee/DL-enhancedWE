#!/bin/bash

# Set up simulation environment
source env.sh

# Clean up from previous/ failed runs
rm -rf traj_segs seg_logs istates west.h5 ddwe-logs
mkdir   seg_logs traj_segs istates

# Note all states from the "clusters" in bstates.txt are from RMSD > 10.2 Ang
# The tstate would be structures with RMSD < 1 Ang

w_init --bstate-file "bstates/bstates.txt" \
        --tstate-file "tstate.file" \
	--segs-per-state 4 \
	--work-manager=threads "$@" > init.log
