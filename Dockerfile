FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the main bot file
COPY eth_options_bot_india_pro.py .

# Expose port
EXPOSE 8000

# Environment variable
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "-u", "eth_options_bot_india_pro.py"]
