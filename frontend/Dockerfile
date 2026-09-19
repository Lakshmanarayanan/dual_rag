FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install Streamlit & Requests
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app listening on all network interfaces
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]