FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY eth_options_bot_india_pro.py .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "eth_options_bot_india_pro.py"]
