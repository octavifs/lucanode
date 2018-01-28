FROM octavifs/simpleelastix_python:ubuntu17.10-python3.6
RUN pip3 install ipyparallel
RUN ipcluster nbextension enable

WORKDIR /notebooks
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--allow-root"]
