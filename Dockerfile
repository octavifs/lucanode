FROM continuumio/miniconda3
# Setup environment
COPY environment-cpu.yml .
RUN conda env update -f ./environment-cpu.yml -n base
# Copy code
COPY . /lucanode
WORKDIR /lucanode
# Define entrypoints
VOLUME ["/input", "/output"]
