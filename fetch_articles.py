
import os
import re
import json
from tavily import TavilyClient

# --- config: edit this list ---
CELEBRITIES = [
    "Taylor Swift",
    "Zendaya",
    "Timothee Chalamet",
    "Selena Gomez",
    "Dwayne Johnson",
]
ARTICLES_PER_PERSON = 8
OUTPUT_PATH = "articles.json"
# --------------------------------

def normalize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def dedup_articles(articles: list[dict], threshold: float = 0.8) -> list[dict]:
    
    kept = []
    kept_tokens = []
    for art in articles:
        tokens = normalize(art["title"])
        is_dup = False
        for existing in kept_tokens:
            union = tokens | existing
            if not union:
                continue
            overlap = len(tokens & existing) / len(union)
            if overlap >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(art)
            kept_tokens.append(tokens)
    return kept


def fetch_for_celebrity(client: TavilyClient, name: str, n: int) -> list[dict]:
    
    try:
        result = client.search(
            query=f"{name} news",
            search_depth="advanced",
            topic="news",
            max_results=n,
            include_raw_content=False,  # the search-result snippet is enough for a weekend project
        )
    except Exception as e:
        print(f"  search failed for {name}: {e}")
        return []

    articles = []
    for r in result.get("results", []):
        if not r.get("content"):
            continue  # skip empty results rather than caching dead weight
        articles.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),          # snippet/summary text
            "published_date": r.get("published_date", ""),
        })
    return dedup_articles(articles)


def main():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise SystemExit("Set TAVILY_API_KEY in your environment first.")

    client = TavilyClient(api_key=api_key)

    dataset = {}
    for name in CELEBRITIES:
        print(f"Fetching articles for {name}...")
        articles = fetch_for_celebrity(client, name, ARTICLES_PER_PERSON)
        if not articles:
            print(f"  WARNING: 0 usable articles for {name} -- skipping "
                  "them (chat script will just have nothing to say)")
            continue
        dataset[name] = articles
        print(f"  got {len(articles)} articles after dedup")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
