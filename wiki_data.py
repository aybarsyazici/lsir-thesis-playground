import requests

def fetch_wikidata_info(item_id, languages=["en", "de", "fr", "uk", "es", "ru", "it"]):
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": 'Q'+str(item_id),
        "format": "json",
        "props": "labels|descriptions|aliases"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        entity = data.get("entities", {}).get('Q'+str(item_id), {})
        
        # Extracting relevant fields
        labels = [label["value"] for label in entity.get("labels", {}).values() if label["language"] in languages]
        description = entity.get("descriptions", {}).get("en", {}).get("value", "")
        all_aliases = []
        for alias_array in entity.get("aliases", {}).values():
            for alias in alias_array:
                if alias["language"] in languages:
                    all_aliases.append(alias["value"])
        
        return {
            "labels": labels,
            "description": description,
            "aliases": all_aliases
        }
    return None

print(fetch_wikidata_info(3874799))