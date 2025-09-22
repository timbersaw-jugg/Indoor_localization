#  test_api.py

import requests
import numpy as np  
import json

# the url of our BentoML service
url = "http://127.0.0.1:3000/predict"

# create a sample input(1 sample, 20 time steps, 13 features)
print("Creating sample input data...")
sample_input = np.random.rand(1, 20, 13).astype(np.float32)

print(f"Sending request to {url}...")

#send the request
response = requests.post(
    url, 
    headers={"Content-Type": "application/json"},
    data=json.dumps({"input_sequence": sample_input.tolist()})
)

#print the response
print("Response from the service:")
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")
