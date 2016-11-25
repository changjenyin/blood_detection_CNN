#!/usr/bin/env sh

# Create and shuffle meta/
python sanitize.py
shuf meta/train.txt > meta/tmp.txt
mv meta/tmp.txt meta/train.txt
#cp meta/* /tmp3/changjenyin/caffe/data
echo 'meta/ Done.'

# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
EXAMPLE=./
DATA=meta/
TOOLS=../caffe/build/tools/
TRAIN_DATA_ROOT=/
VAL_DATA_ROOT=/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

#if [ ! -d "$TRAIN_DATA_ROOT" ]; then
#  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
#  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet training data is stored."
#  exit 1
#fi
#
#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1
#fi

echo "Cleaning existing lmdb, mean..."
rm -rf bloody_train_lmdb bloody_val_lmdb mean.binaryproto mean.npy

echo "Creating train lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/bloody_train_lmdb

echo "Creating val lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/bloody_val_lmdb
echo "lmdb Done."

# Compute the mean image from train_lmdb
/tmp3/changjenyin/caffe/build/tools/compute_image_mean.bin bloody_train_lmdb mean.binaryproto
python convert_bprotonpy.py mean.binaryproto mean.npy
echo ".binaryproto, .npy Done."
