FROM python:3.10-slim
WORKDIR /app

COPY . /app

VOLUME /app/data

RUN pip3 install -r requirements.txt

RUN chmod +x /app/predict.py

# Предскачиваем модель
RUN chmod +x /app/download_model.py && python download_model.py



CMD ["python3","/app/predict.py"]