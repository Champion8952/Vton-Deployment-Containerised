# Base image with CUDA support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install Python, pip, and required libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

# Create a symlink for 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8002

# Command to start the server
CMD ["python", "start.py"]
