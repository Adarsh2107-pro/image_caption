FROM python:3.10-slim

WORKDIR /app

# Copy app file
COPY image_caption/app.py app.py

# Copy models
COPY models/ models/

# Copy Python package folder (your own code)
COPY image_caption/ image_caption/

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Cloud Run port
EXPOSE 8080

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]


