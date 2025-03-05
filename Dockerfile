# syntax=docker/dockerfile:1

FROM python:3.13.1

EXPOSE 8080

WORKDIR /ttds_assignment

COPY requirements.txt requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .
CMD ["python", "back_end/backend.py"]
