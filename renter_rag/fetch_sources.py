"""Download the authorised legislation files listed in sources.yaml (skips files already present).

Run:  python fetch_sources.py
"""

import requests
import yaml

from config import ROOT


def main():
    sources = yaml.safe_load((ROOT / "sources.yaml").read_text())
    for doc in sources["legislation"]:
        path = ROOT / doc["local"]
        if path.exists():
            print(f"have   {doc['title']}  ({path.name})")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        reply = requests.get(doc["file"], headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        reply.raise_for_status()
        path.write_bytes(reply.content)
        print(f"saved  {doc['title']}  ({len(reply.content) // 1024} KB)")


if __name__ == "__main__":
    main()
