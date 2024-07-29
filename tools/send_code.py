import json

def send_code(code):
    print(code)
    return json.dumps({"result": "ok"})