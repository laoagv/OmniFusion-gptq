FROM python:3.12-slim
WORKDIR /usr/src/omnifusion-gptq
RUN apt-get update
RUN apt-get install build-essential -y
RUN pip3 install torch torchvision torchaudio
RUN pip3 install -v gptqmodel --no-build-isolation 
RUN pip3 install transformers[sentencepiece]
RUN pip3 install sentencepiece
RUN pip3 install websockets

COPY . .

EXPOSE 8765
CMD ["python3", "-u", "server.py", "cuda:0"]
