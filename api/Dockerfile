FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY ./ ./

RUN pip install pipreqs
RUN pipreqs --force .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python3", "./main.py"]