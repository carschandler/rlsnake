FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY lib ./lib/
COPY notebooks ./notebooks/
COPY scripts ./scripts/
COPY docker_only .

RUN conda env update --file environment.yml && conda clean --all --yes
