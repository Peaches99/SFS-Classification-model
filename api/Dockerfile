FROM python:3.10.10-slim-bullseye

WORKDIR /app

COPY ./ ./

RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8000

CMD ["python3", "./main.py"]