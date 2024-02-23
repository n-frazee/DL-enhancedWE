#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT

ALIGN=$(mktemp)

COMMAND="           parm $WEST_SIM_ROOT/common_files/hmr.prmtop \n"
COMMAND="${COMMAND} trajin $WEST_STRUCT_DATA_REF \n"
COMMAND="${COMMAND} reference $WEST_SIM_ROOT/common_files/2rvd.pdb [reference] \n"
COMMAND="${COMMAND} rms ALIGN @CA reference out $ALIGN \n"
COMMAND="${COMMAND} go"

#echo -e "${COMMAND}" | $CPPTRAJ

CON=$(mktemp)

python $WEST_SIM_ROOT/common_files/best_hummer.py $CON

cat $CON > $WEST_PCOORD_RETURN

cp $WEST_SIM_ROOT/common_files/hmr.prmtop $WEST_TRAJECTORY_RETURN
cp $WEST_STRUCT_DATA_REF $WEST_TRAJECTORY_RETURN

cp $WEST_SIM_ROOT/common_files/hmr.prmtop $WEST_TRAJECTORY_RETURN
cp $WEST_STRUCT_DATA_REF $WEST_TRAJECTORY_RETURN/parent.ncrst

rm $DIHED

if [ -n "$SEG_DEBUG" ] ; then
  head -v $WEST_PCOORD_RETURN
fi
