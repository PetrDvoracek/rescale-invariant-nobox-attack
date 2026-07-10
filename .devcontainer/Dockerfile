FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG USERNAME=pedro
# TODO run `export USER_EMAIL=...` in terminal before building the container
ARG USER_EMAIL
ARG USER_UID=1000
ARG USER_GID=$USER_UID


COPY requirements.txt /tmp/
RUN apt-get update  \
    && pip install --upgrade "pip<25" \
    # remove all opencv, problem with its import https://github.com/opencv/opencv-python/issues/884
    && pip install opencv-fixer==0.2.5 \
    && python -c "from opencv_fixer import AutoFix; AutoFix()" \
    # && pip uninstall $(pip list --format=freeze | grep opencv) -y \ 
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    # mmdetection 
    && python -m mim install "mmengine==0.10.3" "mmcv==2.1.0" "mmdet==3.3.0" \
    # create non-root user
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # libgl1-mesa-glx
    && apt-get install -y sudo x11-apps libgl1 \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# git config must run as the user, otherwise it lands in /root/.gitconfig
USER $USERNAME
RUN git config --global user.email "${USER_EMAIL}" \
    # git use main instead master branch name
    && git config --global init.defaultBranch main \
    # git avoid dubious ownership
    && git config --global --add safe.directory "*"

