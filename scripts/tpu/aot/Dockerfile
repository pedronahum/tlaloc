# XLA:TPU for compiling without a TPU (scripts/tpu/aot-check.sh). libtpu is
# published for x86_64 only; on an arm64 host the image runs under QEMU.
FROM python:3.12-slim
ARG JAX_VERSION=0.12.0.1
ARG LIBTPU_VERSION=0.0.50.*
RUN pip install --no-cache-dir "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}" "libtpu==${LIBTPU_VERSION}"
ENV TPU_ACCELERATOR_TYPE=v6e-1 TPU_WORKER_HOSTNAMES=localhost
