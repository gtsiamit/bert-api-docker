# Python image
FROM python:3.13-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (code and model)
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "bert_api_docker.server:app", "--host", "0.0.0.0", "--port", "8000"]
