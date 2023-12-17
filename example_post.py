import requests
import json

# Define the URL
url = 'http://localhost:8000/embeddings/'

# Define the headers
headers = {'Content-Type': 'application/json'}

# Define the data
data = {
    'text': 'Hello, world!',
}

# Send the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))

# Print the response
print(response.json())