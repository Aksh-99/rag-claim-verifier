# Celebrity gossip RAG chatbot (weekend build)

## Setup

```
pip install -r requirements.txt
export TAVILY_API_KEY=your_key      # tavily.com, free tier
export ANTHROPIC_API_KEY=your_key
```

## Run

```
python fetch_articles.py   # once, caches articles.json
python rag_chat.py         # chat loop, reads only from articles.json
```

## How it works

`fetch_articles.py` hits the Tavily search API once per celebrity and saves the results to `articles.json`. This is your fixed dataset -- the chat step never searches live, so it's fast and reproducible.

`rag_chat.py` does retrieval with plain keyword overlap (no vector store needed at this scale -- a handful of articles per person). For each question it pulls the top-k most relevant cached articles, stuffs them into the prompt as numbered sources, and asks Claude to answer conversationally while tagging each claim confirmed / reported / rumored based on how many sources back it and whether the source itself hedges.

Multi-turn chat: the running conversation is kept in history, but only the latest turn gets the full source dump -- earlier turns just keep the plain question/answer text, so the context window doesn't balloon with retrieved articles from three questions ago.

## Known limitations (deliberately not fixed)

- No real cross-source fact verification -- the confirmed/reported/rumored tag is still a heuristic (source count + hedging language), not verified truth. The system prompt tells the model to default to "reported" when unsure rather than let its own background knowledge of the celebrity push a claim up a tier -- but that's a prompt instruction, not a check. It can still get this wrong on thin 2-3 sentence snippets.
- Retrieval is keyword-overlap (now stopword-filtered), not semantic -- "who is she dating" and "relationship history" still won't match well despite being related. Good enough for ~40 cached articles, not a substitute for embeddings at any real scale.
- No UI -- this is a CLI loop. Swapping in Streamlit is maybe 30 min if you want a demo-able front end later.

## Fixed after review

- **Stopword filtering in retrieval** -- "who/is/the/a" no longer inflate the overlap score on short questions.
- **Grounding across turns** -- previously, source snippets were stripped from conversation history after each turn, so "wait, which source said that?" two turns later had nothing to resolve against. Now a `SourceRegistry` assigns each article a stable number the first time it's retrieved and keeps a compact title/url reference in history for every turn, so later questions can still resolve "Source 4" to a real citation without re-sending full snippet text every time.
- **Dedup** -- near-identical titles (common with syndicated entertainment news) are dropped before caching, so the same wire story reposted on three sites doesn't inflate a claim to "confirmed" by looking like independent corroboration. Same failure mode your citation-graph work needs to handle for real, just simpler here.
- **Basic error handling** -- missing/empty/corrupt `articles.json`, unmatched celebrity name, failed Tavily calls, failed Anthropic calls. Still not production-grade, but it won't just stack-trace on the first hiccup.
