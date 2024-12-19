# Use the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# NVIDIA drivers and CUDA toolkit should be installed on the host machine
# The container will use the host's NVIDIA drivers through nvidia-container-toolkit
# We only need the CUDA runtime libraries in the container
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-runtime-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Check if the driver is installed     
RUN nvidia-smi

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8002

# Command to start the server
CMD ["python", "start.py"]
