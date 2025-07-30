#!/bin/bash

# get the path to the current directory
MY_PATH=`dirname "$0"`
MY_PATH=`( cd "$MY_PATH" && pwd )`
cd $MY_PATH

cd images
rm *.sif

ARCH=$( uname -m )

if [[ $ARCH == "aarch64" ]]; then
  wget -c  https://nasmrs.felk.cvut.cz/index.php/s/JOjd3JQWcN4lwED/download -O mrs_uav_system.sif --no-check-certificate
else
   wget -c https://nasmrs.felk.cvut.cz/index.php/s/zoUKhkxgvDmIppY/download -O mrs_uav_system.sif --no-check-certificate
fi
