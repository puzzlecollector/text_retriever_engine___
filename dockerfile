FROM nvcr.io/nvidia/pytorch:20.07-py3

RUN mkdir /root/textretriever

COPY . /root/textretriever

WORKDIR /root/textretriever

RUN python -m pip --no-cache-dir install --upgrade pip grpcio grpcio-tools && \
    ldconfig && \
    rm -rf /tmp/* /workspace/* && \
    python -m grpc.tools.protoc -I. --python_out . --grpc_python_out . ./textretriever.proto

EXPOSE 35015

ENTRYPOINT ["python", "textretriever_server.py"]
