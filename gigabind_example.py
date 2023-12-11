import modal
from modal import Client

server_url = "http://localhost:5000"  # replace with your server URL
client_type = "example_type"  # replace with your client type
credentials = {"username": "example", "password": "example"}  # replace with your credentials

client = Client(server_url=,client_type=,credentials=)
result = client.call("gigabind", {"input_data": {"text": ["Hello, world!"]}})
print(result)

