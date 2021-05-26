FROM python:3.9

WORKDIR /flair

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . /flair
