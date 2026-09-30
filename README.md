# pick-and-roll

A retrieval pipeline where a small local LLM picks operations and deterministic code does the counting, filtering, and citing.

## The problem

A 2B-parameter model answers lookups fine but fails at sums, maxes, and "first game" logic. Similarity search alone can't do those either.

## How it works

1. **Resolve.** Match teams, players, and dates in the question against the tables. No LLM involved, so names can't be invented.
2. **Plan.** The model picks operations from a fixed menu (lookup, sum, count, max, first, last). Code fills in every entity and date.
3. **Execute.** pandas runs the operation over the joined tables and returns the facts plus the exact rows used as evidence.
4. **Text fallback.** For questions that live in recaps or injury notes, the tables pin the game first. Then embedding retrieval picks among the candidates, and the pipeline abstains if the top two are too close.

## Design choices

- **Abstain over guess.** If a required row doesn't exist, the pipeline says so instead of returning a confident number.
- **Tables win.** Text sources can be wrong, so every score comes from the structured tables.
- **Full-name matching.** Two players can share a surname, so fuzzy matching is deliberately avoided.

## Status

This repo holds only my modules. It was built on a provided starter and dataset that I can't share, so it isn't runnable as is.

## Stack

Python, pandas, SQLAlchemy, pgvector, Ollama.
