import json

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    