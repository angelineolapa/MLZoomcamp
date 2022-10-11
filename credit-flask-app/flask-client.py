#Script to run flask app
import requests ## to use the POST method we use a library named requests

url = 'http://localhost:9696/predict' ## this is the route we made for prediction
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
requests.post(url, json=client).json()
print(result)