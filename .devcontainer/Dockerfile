FROM ghcr.io/mamba-org/micromamba:bullseye

####################################
USER root
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential pkg-config libssl-dev lldb curl vim \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Node.js

# RUN curl -s https://deb.nodesource.com/setup_18.x | \
#     sudo bash - && apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get install -y nodejs && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# RUN npm install -g sass

# Add vscode user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server*

# Add sudo support for the vscode user
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y sudo \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install zshrc
COPY .devcontainer/.zshrc /home/vscode/.zshrc
RUN chown ${USER_UID}:${USER_GID} /home/vscode/.zshrc

####################################
# USER root

# COPY environment.yml* .devcontainer/noop.txt /tmp/mamba-tmp/
# RUN if [ -f "/tmp/mamba-tmp/environment.yml" ]; then umask 0022 && micromamba install -y -n base -f /tmp/mamba-tmp/environment.yml && \
#     micromamba clean --all --yes; fi \
#     && rm -rf /tmp/mamba-tmp

####################################
USER vscode

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly

ENV PATH="/home/vscode/.cargo/bin:${PATH}"

RUN rustup update

RUN rustup component add clippy rustfmt rust-src rust-analysis rls 

RUN cargo install cargo-edit cargo-deny cargo-audit cargo-update cargo-machete eza just

ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Init micromamba
#ENV MAMBA_ROOT_PREFIX=/opt/micromamba
#ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
# ENV MAMBA_EXE=/bin/micromamba

# RUN micromamba shell init -s zsh -p /opt/micromamba && \
#RUN echo "source /opt/micromamba/etc/profile.d/mamba.sh" >> /home/vscode/.zshrc
