# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run gigabind.py when the container launches
CMD ["python", "gigabind.py"]
