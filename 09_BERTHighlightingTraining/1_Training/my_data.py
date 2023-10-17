import json

json_file = open("all_highlighted_tickets.json", encoding="utf8")
data = json.load(json_file)
json_file.close()
def get_data():
    return data