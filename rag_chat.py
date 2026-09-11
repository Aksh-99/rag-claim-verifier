"""
Step 2: Chat with a celebrity-gossip RAG bot, grounded in the articles
cached by fetch_articles.py. No live search here -- retrieval happens
over the fixed local dataset.

Requires: ANTHROPIC_API_KEY in your environment.
"""

import os
import json
import re
from anthropic import Anthropic
from fetch_articles import fetch_for_celebrity, TavilyClient

DATA_PATH = "articles.json"
MODEL = "claude-sonnet-4-6"
TOP_K = 5  # how many articles to retrieve per question


def save_dataset(dataset: dict):
    with open(DATA_PATH, "w") as f:
        json.dump(dataset, f, indent=2)


def fetch_new_celebrity(name: str) -> list[dict]:
    """Live fallback for a name that isn't in the cached dataset yet. Only
    runs once per new name -- the result gets saved back to articles.json
    afterward, so next time this celebrity behaves like a pre-fetched one
    (no live call needed)."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("  Can't fetch live -- TAVILY_API_KEY isn't set.")
        return []
    client = TavilyClient(api_key=api_key)
    return fetch_for_celebrity(client, name, n=8)


def load_dataset() -> dict:
    """Load the cache if it exists. Missing or empty is fine now -- with
    live search, articles.json is just a growing cache, not a required
    pre-fetch step."""
    if not os.path.exists(DATA_PATH):
        return {}
    try:
        with open(DATA_PATH) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise SystemExit(
            f"{DATA_PATH} exists but isn't valid JSON -- delete it and try again."
        )
    return data or {}


# Small stopword list -- enough to stop "who/is/the/a" from swamping the
# score on short questions. Not exhaustive, but covers the common offenders.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "who", "what", "when", "where", "why", "how", "which", "does", "did",
    "do", "has", "have", "had", "to", "of", "in", "on", "for", "and",
    "or", "with", "about", "her", "his", "she", "he", "they", "them",
    "it", "its", "this", "that", "there", "as", "at", "by", "from",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def retrieve(question: str, articles: list[dict], k: int) -> list[dict]:
    """Rank articles by stopword-filtered keyword overlap with the question
    and return the top k. Simple, no embeddings needed -- fine at this
    dataset size. Swap this function out for a vector similarity search
    later if you want it to generalize past a few hundred articles.

    Note: this still only matches on shared vocabulary, not meaning --
    "who is she dating" and "relationship history" won't overlap much
    even though a human would consider them related. Fine for a weekend
    demo, a real limitation if you lean on this pattern later."""
    q_tokens = tokenize(question)
    scored = []
    for art in articles:
        art_tokens = tokenize(art["title"] + " " + art["content"])
        overlap = len(q_tokens & art_tokens)
        scored.append((overlap, art))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    # if nothing overlaps at all, just fall back to the most recent articles
    top = [art for score, art in scored if score > 0][:k]
    if not top:
        top = articles[:k]
    return top


def build_context(articles: list[dict], numbers: list[int]) -> str:
    blocks = []
    for i, art in zip(numbers, articles):
        blocks.append(
            f"[Source {i}] {art['title']} ({art.get('published_date') or 'date unknown'})\n"
            f"{art['content']}\n"
            f"URL: {art['url']}"
        )
    return "\n\n".join(blocks)


class SourceRegistry:
    """Assigns a stable number to each article the first time it's
    retrieved, and remembers it for the rest of the conversation -- so
    when the user asks 'wait, which source said that?' two turns later,
    the model (and you) can still resolve 'Source 4' to a real title/url,
    even though we don't re-send the full snippet text every turn."""

    def __init__(self):
        self.url_to_number = {}
        self.entries = {}  # number -> {title, url}

    def assign(self, articles: list[dict]) -> list[int]:
        numbers = []
        for art in articles:
            url = art["url"]
            if url not in self.url_to_number:
                n = len(self.url_to_number) + 1
                self.url_to_number[url] = n
                self.entries[n] = {"title": art["title"], "url": url}
            numbers.append(self.url_to_number[url])
        return numbers

    def compact_reference(self, numbers: list[int]) -> str:
        lines = [f"Source {n}: {self.entries[n]['title']} ({self.entries[n]['url']})"
                 for n in numbers]
        return "\n".join(lines)


SYSTEM_PROMPT = """You are a chatty, well-informed entertainment-news assistant. \
You answer questions ONLY using the numbered sources provided in each message -- \
never use outside knowledge about the celebrity, even if you're confident \
you know the answer. If you catch yourself about to state something because \
you recall it rather than because a numbered source says it, don't.

Source snippets are short (2-3 sentences), which makes real corroboration \
hard to judge. Be conservative:
- (confirmed) -- ONLY if 2+ of the given sources state the same specific \
claim, not just the same general topic
- (reported) -- stated as fact by a single source, not corroborated
- (rumored) -- the source itself uses hedging language ("reportedly", \
"sources say", "rumored to be", "according to insiders", etc.)
When in doubt between confirmed and reported, use reported -- do not let \
your own background knowledge of the celebrity push a claim up a tier.

If the sources don't cover something the user asks, say so plainly instead \
of guessing. Cite sources by number, e.g. "according to Source 2".

Tone: talk like a gen z girl texting her bestie the tea -- casual, \
enthusiastic, lowercase-leaning energy is fine, slang like "no because", \
"the way that", "she is NOT", "ok wait" is fine, the occasional emoji is \
fine. But don't let the voice water down the substance: still hit the \
confirmed/reported/rumored tags on every claim and still cite sources by \
number. Gossipy delivery, accurate content."""


def chat_loop():
    dataset = load_dataset()
    names = list(dataset.keys())

    choice = input("okay bestie who we talking about?? 👀 ").strip()
    if not choice:
        print("say a name lol\n")
        return chat_loop()

    # cache hit -- exact or partial match against what we already have,
    # so re-asking about someone doesn't trigger a fresh live fetch
    celeb = None
    matches = [n for n in names if choice.lower() in n.lower() or n.lower() in choice.lower()]
    if matches:
        celeb = matches[0]

    # cache miss -- search live, no confirmation needed since this is
    # meant to feel like a search box, not a gate
    if celeb is None:
        print(f"one sec, digging up the tea on {choice}...")
        articles = fetch_new_celebrity(choice)
        if articles:
            dataset[choice] = articles
            save_dataset(dataset)
            celeb = choice
            print(f"  ok found {len(articles)} articles, saving for next time 📌")
        else:
            print(
                f"  couldn't find anything on '{choice}' 😭 "
                "(check TAVILY_API_KEY is set, or maybe check the spelling)\n"
            )
            return chat_loop()

    print(f"\nspilling the tea on {celeb}. type 'quit' to bounce, 'switch' for someone new.\n")

    try:
        client = Anthropic()  # picks up ANTHROPIC_API_KEY from env
    except Exception as e:
        raise SystemExit(f"Couldn't set up the Anthropic client: {e}")

    history = []
    registry = SourceRegistry()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "switch":
            return chat_loop()

        articles = retrieve(question, dataset[celeb], TOP_K)
        numbers = registry.assign(articles)
        context = build_context(articles, numbers)

        user_turn = f"Sources:\n\n{context}\n\nQuestion: {question}"
        history.append({"role": "user", "content": user_turn})

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=history,
            )
        except Exception as e:
            print(f"\n[API error, try again: {e}]\n")
            history.pop()  # don't leave a dangling unanswered turn in history
            continue

        answer = response.content[0].text
        print(f"\nBot: {answer}\n")

        # Keep the visible conversation lean but NOT ungrounded: store the
        # question plus a compact title/url reference for the sources used
        # this turn (not the full snippet text), so a later "which source
        # said that?" can still be resolved instead of hitting a dead end.
        reference = registry.compact_reference(numbers)
        history[-1] = {
            "role": "user",
            "content": f"{question}\n\n(Retrieved from:\n{reference})",
        }
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    chat_loop()