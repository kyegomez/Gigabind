import requests
import json

# replace with your server URL
server_url = "http://localhost:5000/process"  
# replace with your API key
api_key = "your_api_key"  

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Example for text modality
input_data = {"text": ["Hello, world!"]}
response = requests.post(server_url, headers=headers, data=json.dumps({"input_data": input_data}))
print(response.json())

# Example for audio modality
# Assuming audio_file is a file object opened in binary mode
with open("path_to_your_audio_file", "rb") as audio_file:
    files = {"audio": audio_file}
    response = requests.post(server_url, headers=headers, files=files)
    print(response.json())

# Example for vision modality
# Assuming image_file is a file object opened in binary mode
with open("path_to_your_image_file", "rb") as image_file:
    files = {"vision": image_file}
    response = requests.post(server_url, headers=headers, files=files)
    print(response.json())

