# Use an official Python image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Copy supervisor config into container
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy all files into /app (this includes backend, frontend, models, etc.)
COPY . /app

# Install backend dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install frontend dependencies
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Final command: launch supervisor, which starts both backend & frontend
CMD ["/usr/bin/supervisord"]
