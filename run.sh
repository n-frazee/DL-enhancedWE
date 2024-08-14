#!/bin/bash

# Make sure environment is set
source env.sh

# Clean up
rm -f west.log

# Run w_run
#w_run "$@" > west.log
w_run --n-workers 1 "$@" |tee west.log
