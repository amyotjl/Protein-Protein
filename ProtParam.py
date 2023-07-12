import requests
import re


URL = 'https://web.expasy.org/cgi-bin/protparam/protparam'
def get_sequence_half_life( sequence, url=URL):
    payload = {
        'sequence' : sequence
    }
    response = requests.post(url, payload)

    if response.status_code == 200:

        html  = response.text

        # extract the half life from html 
        pattern = r"half-life is:\s?(\d+(\.\d+)?) hours"
        match = re.search(pattern, html)
        if match:
            return match.group(1) 
        else:
            return None
    else :
        print("Request failed with status code:", response.status_code)
        return None
    