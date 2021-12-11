#!/bin/sh
mkdir -p datasets

########################
###      Replica     ###   https://github.com/facebookresearch/Replica-Dataset
########################
#if [ -d "datasets/replica" ]; then
#  echo "Replica dataset already downloaded!"
#else
#  echo "Downloading Replica dataset..."
#  ./subrepos/Replica-Dataset/download.sh datasets/replica
#  echo "Replica dataset download complete!"
#fi

########################
###      EDE20K      ###  https://groups.csail.mit.edu/vision/datasets/ADE20K/
########################
#if [ -d "datasets/ADE20K" ]; then
#  echo "ADE20K dataset already downloaded!"
#else
#  echo "Downloading EDE20K dataset..."
#  mkdir -p datasets/ADE20K
#  wget --continue -P datasets/ADE20K https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
#  unzip -d datasets/ADE20K datasets/ADE20K/ADE20K_2016_07_26.zip
#  echo "ADE20K dataset download complete!"
#fi

########################
###      CamVid      ###
########################
# Source: https://github.com/alexgkendall/SegNet-Tutorial
if [ -d "datasets/CamVid" ]; then
  echo "CamVid dataset already downloaded!"
else
  echo "Downloading CamVid dataset..."
  wget --continue -P datasets https://github.com/alexgkendall/SegNet-Tutorial/archive/master.zip
  unzip -d datasets datasets/master.zip
  mv datasets/SegNet-Tutorial-master/CamVid datasets
  rm datasets/master.zip
  rm -rf datasets/SegNet-Tutorial-master
  echo "CamVid dataset download complete!"
fi

########################
###      KITTI       ###
########################
# Source: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015
if [ -d "datasets/kitti" ]; then
  echo "KITTI dataset already downloaded!"
else
  echo "Downloading KITTI dataset..."
  command -v gdown >/dev/null 2>&1 || { echo >&2 "I require 'gdown' command, but it's not installed (pip install --user gdown). Aborting!"; exit 1; }
  gdown --id 1FFzxWEcjRz4g1Oy7uAKh-ZHcvq-N9U9Q --output datasets/kitti.zip
  unzip -d datasets datasets/kitti.zip
  rm datasets/kitti.zip
  echo "KITTI dataset download complete!"
fi
