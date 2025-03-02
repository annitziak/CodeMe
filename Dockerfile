# syntax=docker/dockerfile:1

FROM python:3.13.1

EXPOSE 8080

WORKDIR /ttds_assignment

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .
CMD [ "python", "-m" , "flask", "--app","back_end/backend","run","--host=0.0.0.0","--port=8080"]
