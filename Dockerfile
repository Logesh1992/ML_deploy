# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy the requirements file into the container at /code
COPY ./requirements.txt /code/requirements.txt

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of the application code into the container at /code
COPY ./app /code/app
COPY ./html.html /code/html.html  
COPY ./static /code/static

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
