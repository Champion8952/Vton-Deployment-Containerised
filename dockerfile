# Use the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (ensure it matches `settings.PORT`)
EXPOSE 8002

# Command to start the server
CMD ["python", "start.py"]
