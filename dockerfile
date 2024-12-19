# Use the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Check if the driver is installed     
# RUN nvidia-smi

# Install the driver
RUN apt-get update && apt-get install -y nvidia-driver-552

# Install the toolkit
RUN apt-get update && apt-get install -y nvidia-cuda-toolkit

# Check if the toolkit is installed
RUN nvidia-smi

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8002

# Command to start the server
CMD ["python", "start.py"]
