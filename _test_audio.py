import requests
import json

files_to_test = [
    "Recording 2026-03-02 175722 (1).mp3",
    "Recording 2026-03-02 175722.mp3",
    "Voicy_Ye college hai pressure cooker nae.mp3"
]

for fname in files_to_test:
    try:
        with open(fname, "rb") as f:
            resp = requests.post("http://localhost:8081/verify", files={"file": (fname, f, "audio/mpeg")})
        data = resp.json()
        print(f"File      : {fname}")
        print(f"Predicted : {data.get('speaker') or data.get('closest_speaker')}")
        print(f"Similarity: {data.get('similarity')}")
        print(f"Identified: {data.get('identified')}")
        top = data.get("top_matches", [])
        for i, m in enumerate(top):
            print(f"  Top{i+1}: {m['speaker']} = {m['similarity']}")
        print()
    except Exception as e:
        print(f"ERROR on {fname}: {e}")
        print()
