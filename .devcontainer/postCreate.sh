#!/bin/bash

sudo apt-get install -y tmux tree

bash .devcontainer/zsh-in-docker.sh \
    -t amuse \
    -p git \
    -p extract \
    -p copybuffer
