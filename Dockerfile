FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y tzdata
RUN apt-get install -y python3-pip
RUN apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs
RUN pip3 install numpy scipy scikit-image jupyterlab seaborn pandas lime
RUN pip3 install scikit-learn==0.24.0
RUN pip3 install tensorflow==2.4.0
RUN pip3 install plotly
RUN pip3 install ipywidgets
RUN jupyter labextension install jupyterlab-plotly

EXPOSE 8888

CMD [ "jupyter", "lab", "--notebook-dir=/opt/notebooks", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root" ]
