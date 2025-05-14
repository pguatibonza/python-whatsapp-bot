# 1. Use a slim Python base
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# 5. Copy your application code
COPY . .

# 6. Tell Docker/Podman that the container listens on 8080
EXPOSE 8080

# 7. Launch with Hypercorn binding to 0.0.0.0:8080
CMD ["hypercorn", "run:app", "--bind", "0.0.0.0:8080"]
