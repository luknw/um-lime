FROM ubuntu:20.04

RUN apt-get update && \
  apt-get install -y tzdata && \
  apt-get install -y python3-pip && \
  apt-get install -y nodejs
RUN pip3 install numpy scipy scikit-image jupyterlab seaborn pandas lime 
RUN pip3 install scikit-learn==0.24.0
RUN pip3 install tensorflow==2.4.0
RUN pip3 install plotly


EXPOSE 8888

CMD [ "jupyter", "lab", "--notebook-dir=/opt/notebooks", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root" ]
