#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

ln -sv $WEST_SIM_ROOT/common_files/hmr.prmtop .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.in > md.in
  ln -sv $WEST_PARENT_DATA_REF/seg.ncrst ./parent.ncrst
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.in > md.in
  ln -sv $WEST_PARENT_DATA_REF ./parent.ncrst
fi

$PMEMD -O -i md.in  -p hmr.prmtop  -c parent.ncrst \
          -r seg.ncrst -x seg.nc      -o seg.log    -inf seg.nfo

ALIGN=$(mktemp)
CON=$(mktemp)

COMMAND="           parm hmr.prmtop \n"
COMMAND="${COMMAND} trajin $WEST_CURRENT_SEG_DATA_REF/parent.ncrst\n"
COMMAND="${COMMAND} trajin $WEST_CURRENT_SEG_DATA_REF/seg.nc\n"
COMMAND="${COMMAND} reference $WEST_SIM_ROOT/common_files/ref.ncrst [reference] \n"
COMMAND="${COMMAND} rms ALIGN @CA reference out $ALIGN \n"
COMMAND="${COMMAND} radgyr RADGYR @CA,C,O,N,H  out $ALIGN mass nomax \n"
COMMAND="$COMMAND vector D1 :10&!@H= :5,6&!@H= \n"
COMMAND="$COMMAND vector D2 :1&!@H= :5,6&!@H= \n"
COMMAND="$COMMAND run \n"
COMMAND="$COMMAND writedata angle.nc vectraj D1 D2 trajfmt netcdf parmout angle.parm7 \n"
COMMAND="$COMMAND vectormath vec1 D1 vec2 D2 out $ALIGN name angle dotangle \n"
COMMAND="${COMMAND} go"

echo -e "${COMMAND}" | $CPPTRAJ
cat $ALIGN
python $WEST_SIM_ROOT/common_files/best_hummer.py $CON

cat $ALIGN | tail -n +2 | awk '{print $2}' > $WEST_RMSD_RETURN
cat $CON > $WEST_PCOORD_RETURN
cat $ALIGN | tail -n +2 | awk '{print $3}' > $WEST_RADGYR_RETURN
cat $ALIGN | tail -n +2 | awk '{print $4}' > $WEST_ANGLE_RETURN


python $WEST_SIM_ROOT/common_files/get_coord.py

cat coords.txt > $WEST_RCOORD_RETURN

cp hmr.prmtop $WEST_TRAJECTORY_RETURN
cp seg.nc $WEST_TRAJECTORY_RETURN

cp hmr.prmtop $WEST_RESTART_RETURN
cp seg.ncrst  $WEST_RESTART_RETURN/parent.ncrst

cp seg.log $WEST_LOG_RETURN

# Clean Up
rm $ALIGN coord.npy hmr.prmtop

if [ -n "$SEG_DEBUG" ] ; then
  head -v $WEST_PCOORD_RETURN
fi


