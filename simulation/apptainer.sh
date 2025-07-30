#!/bin/bash

set -e

trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "$0: \"${last_command}\" command failed with exit code $?"' ERR

## | ------------------- configure the paths ------------------ |

# Change the following paths when moving the script and the folders around

# get the path to the current directory
MRS_APPTAINER_PATH=`dirname "$0"`
MRS_APPTAINER_PATH=`( cd "$MRS_APPTAINER_PATH" && pwd )`

# alternatively, set it directly
# MRS_APPTAINER_PATH=$HOME/git/mrs_apptainer

# define paths to the subfolders
IMAGES_PATH="$MRS_APPTAINER_PATH/images"
OVERLAYS_PATH="$MRS_APPTAINER_PATH/overlays"
MOUNT_PATH="$MRS_APPTAINER_PATH/mount"

## | ----------------------- user config ---------------------- |

# use <file>.sif for normal container
# use <folder>/ for sandbox container
CONTAINER_NAME="mrs_uav_system.sif"
OVERLAY_NAME="mrs_uav_system.img"

CONTAINED=true  # true: will isolate from the HOST's home
CLEAN_ENV=true # true: will clean the shell environment before runnning container

USE_NVIDIA=false # true: will tell Apptainer that it should use nvidia graphics. Does not work every time.

# the following are mutually exclusive
OVERLAY=false  # true: will load persistant overlay (overlay can be created with scripts/create_overlay.sh)
WRITABLE=false # true: will run it as --writable (works with --sandbox containers, image can be converted with scripts/convert_sandbox.sh)

# defines what should be mounted from the host to the container
# [TYPE], [SOURCE (host)], [DESTINATION (container)]
# - !!! the folders are not being mounted when running with sudo
MOUNTS=(
  # mount the custom user workspace into the container
  #           HOST PATH                                  CONTAINER PATH
  "type=bind" "$MRS_APPTAINER_PATH/.." "/home/$USER/summer-school-2025"

  # mount the MRS shell additions into the container, DO NOT MODIFY
  "type=bind" "$MOUNT_PATH" "/opt/mrs/host"

  # mount folders to facilitate Xserver piping
  "type=bind" "/tmp/.X11-unix" "/tmp/.X11-unix"
  "type=bind" "/dev/dri" "/dev/dri"
  "type=bind" "$HOME/.Xauthority" "/home/$USER/.Xauthority"
)

## | ------------------ advanced user config ------------------ |

# not supposed to be changed by a normal user
DEBUG=false           # true: print the apptainer command instead of running it
KEEP_ROOT_PRIVS=false # true: let root keep privileges in the container
DETACH_TMP=true       # true: do NOT mount host's /tmp
FAKEROOT=false        # true: emulate root inside the container

## | --------------------- user config end -------------------- |

# check whether we are already in tmux
if [ -n "$TMUX" ]; then
  echo "You are probably in a Tmux session. Exit your Tmux first please to avoid nesting multiple Tmux sessions."
  exit 1
fi

if [ -z "$1" ]; then
  ACTION="run"
else
  ACTION=${1}
fi

IMAGE_PATH=$IMAGES_PATH/$CONTAINER_NAME

if $OVERLAY; then

  if [ ! -e $OVERLAYS_PATH/$OVERLAY_NAME ]; then
    echo "Overlay file does not exist, initialize it with the 'create_fs_overlay.sh' script"
    exit 1
  fi

  OVERLAY_ARG="-o $OVERLAYS_PATH/$OVERLAY_NAME"
  $DEBUG && echo "Debug: using overlay"
else
  OVERLAY_ARG=""
fi

if $CONTAINED; then
  CONTAINED_ARG="--home /tmp/apptainer/home:/home/$USER"
  $DEBUG && echo "Debug: running as contained"
else
  CONTAINED_ARG=""
fi

if $WRITABLE; then
  WRITABLE_ARG="--writable"
  FAKEROOT=true
  $DEBUG && echo "Debug: running as writable"
else
  WRITABLE_ARG=""
fi

if $KEEP_ROOT_PRIVS; then
  KEEP_ROOT_PRIVS_ARG="--keep-privs"
  $DEBUG && echo "Debug: keep root privs"
else
  KEEP_ROOT_PRIVS_ARG=""
fi

if $FAKEROOT; then
  FAKE_ROOT_ARG="--fakeroot"
  $DEBUG && echo "Debug: fake root"
else
  FAKE_ROOT_ARG=""
fi

if $CLEAN_ENV; then
  CLEAN_ENV_ARG="-e"
  $DEBUG && echo "Debug: clean env"
else
  CLEAN_ENV_ARG=""
fi

# if we want nvidia, add the "--nv" arg
if $USE_NVIDIA; then
  NVIDIA_ARG="--nv"
else
  NVIDIA_ARG=""
fi

if $DETACH_TMP; then
  TMP_PATH="/tmp/apptainer/tmp"
  DETACH_TMP_ARG="--bind $TMP_PATH:/tmp"
  $DEBUG && echo "Debug: detaching tmp from the host"
else
  DETACH_TMP_ARG=""
fi

if $DEBUG; then
  EXEC_CMD="echo"
else
  EXEC_CMD="eval"
fi

MOUNT_ARG=""
if ! $WRITABLE; then

  # prepare the mounting points, resolve the full paths
  for ((i=0; i < ${#MOUNTS[*]}; i++));
  do
    ((i%3==0)) && TYPE[$i/3]="${MOUNTS[$i]}"
    ((i%3==1)) && SOURCE[$i/3]="${MOUNTS[$i]}"
    ((i%3==2)) && DESTINATION[$i/3]="${MOUNTS[$i]}"
  done

  for ((i=0; i < ${#TYPE[*]}; i++)); do

    if test -e ${SOURCE[$i]}; then

      FULL_SOURCE=$( realpath -e ${SOURCE[$i]} )
      FULL_DESTINATION=$( realpath -m ${DESTINATION[$i]} )

      ADD_MOUNT_ARG="--mount ${TYPE[$i]},source=$FULL_SOURCE,destination=$FULL_DESTINATION"

      $DEBUG && echo $ADD_MOUNT_ARG

      MOUNT_ARG="$MOUNT_ARG $ADD_MOUNT_ARG"

    else

      echo "Error while mounting '${SOURCE[$i]}', the path does not exist".

    fi

  done

fi

if [[ "$ACTION" == "run" ]]; then
  [ ! -z "$@" ] && shift
  CMD="$@"
elif [[ $ACTION == "exec" ]]; then
  shift
  CMD="/bin/bash -c '${@}'"
elif [[ $ACTION == "shell" ]]; then
  CMD=""
else
  echo "Action is missing"
  exit 1
fi

# create tmp folder for apptainer in host's tmp
[ ! -e /tmp/apptainer/tmp ] && mkdir -p /tmp/apptainer/tmp
[ ! -e /tmp/apptainer/home ] && mkdir -p /tmp/apptainer/home

# this will set $DISPLAY in the container to the same value as on your host machine
export APPTAINERENV_DISPLAY=$DISPLAY

xhost + > /dev/null 2>&1

cd ~

$EXEC_CMD apptainer $ACTION \
  $NVIDIA_ARG \
  $OVERLAY_ARG \
  $CONTAINED_ARG \
  $WRITABLE_ARG \
  $CLEAN_ENV_ARG \
  $FAKE_ROOT_ARG \
  $KEEP_ROOT_PRIVS_ARG \
  $MOUNT_ARG \
  $DETACH_TMP_ARG \
  $IMAGE_PATH \
  $CMD

xhost - > /dev/null 2>&1
