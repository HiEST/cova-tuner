FROM python:3.7-slim as build

RUN pip install opencv-python-headless pandas imutils boto3 tqdm sagemaker requests pillow

FROM python:3.7-slim

RUN mkdir /app /edgeautotuner

COPY --from=build /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY src setup.py /edgeautotuner/
RUN cd /edgeautotuner && \
    pip install -e . --no-dependencies

COPY aws/source_dir aws/config.ini aws/*.py /app/
WORKDIR /app


ENTRYPOINT ["/usr/local/bin/python", "e2e-aws.py"]
