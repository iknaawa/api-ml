FROM python:3.10-slim-buster

WORKDIR /app

COPY requirement.txt .  

RUN pip install --no-cache-dir -r requirement.txt  

COPY . .  

EXPOSE 8000  

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]  