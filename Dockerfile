FROM python:3.9-bookworm

WORKDIR /usr/src/app

COPY ./requirements.txt ./

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip --disable-pip-version-check install --no-cache-dir --no-compile  -r requirements.txt

COPY ./README.md ./ \
     ./src ./ \
     ./misc ./

CMD ["python", "src/inference.py"]
