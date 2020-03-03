import requests
import json

url = 'http://0.0.0.0:5000/api/prediction'

data = 'ciao sono un ricercatore'
j_data = json.dumps(data)
# headers = {'content-type': 'applicatclearion/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data)#, headers=headers)
print(r, r.text)