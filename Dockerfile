FROM continuumio/miniconda3

RUN mkdir -p nn-basics

COPY . /nn-basics
WORKDIR /nn-basics

RUN apt-get update && apt-get install -y doxygen graphviz git

RUN conda env create --name nn-basics --file environment.yml

RUN echo "conda activate nn-basics" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
