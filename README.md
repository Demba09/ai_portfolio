# AI Portfolio

**[Try the live demo](https://aiportfolio.streamlit.app/)**

Three applied-AI demonstrators in a single Streamlit app. Each one answers a
concrete business need: query a contractual document, triage inbound requests,
explore data without writing a query.

The user interface is in French. The app runs on real data shipped with the
repository, and starts locally with one command.

```bash
streamlit run app.py
```

---

## The three demos

| Tab | Problem it addresses | Technical core |
|---|---|---|
| **1. Document analysis (RAG)** | Find a specific fact inside a multi-page tender document | Embeddings + FAISS + GPT-4o, sourced answers |
| **2. Support automation** | Qualify an inbound email without a human reading it | GPT-4o + validated Pydantic schema |
| **3. Self-service analytics** | Turn a plain-language question into a chart | Keyword rules + pandas + Plotly |

---

### 1. Document analysis (RAG)

Ask a natural-language question about a PDF and get a **cited** answer, with
the source passages and their page numbers.

The demo document is a real French public-procurement notice published on
BOAMP, covering a data-management contract. Any other PDF can be uploaded
instead.

**Pipeline**

1. Page-by-page text extraction with `pypdf`
2. Chunking into 1200-character segments with a 200-character overlap
3. Vectorisation with `text-embedding-3-small`, normalised vectors
4. FAISS inner-product index, equivalent to cosine similarity on unit vectors
5. Retrieval of the five passages closest to the question
6. Answer generation by GPT-4o, constrained to the retrieved passages only

**The part that matters: the model is allowed not to answer.** The system
prompt requires it to answer strictly from the retrieved passages, and to
return an explicit "information not found in the document" when the answer is
absent. Every factual statement carries a `[Cxx]` citation tied to a page.
On a contractual or regulatory document, an invented answer costs more than
no answer at all.

The passages used are displayed below the answer, so every claim can be
checked against its source.

---

### 2. Support automation (structured extraction)

Turns a support email into a record that can be routed or prioritised.

The model returns JSON validated against a Pydantic schema:

| Field | Type | Allowed values |
|---|---|---|
| `sentiment` | enum | `Très en colère`, `En colère`, `Neutre`, `Satisfait` |
| `urgence` | integer | 1 to 5 |
| `categorie` | enum | `Matériel`, `Logiciel`, `Accès / Identité`, `Réseau`, `Sécurité`, `Demande de service`, `Autre` |
| `action_immediate` | string | non-empty |

Two guardrails sit around the model output. A regular expression isolates the
JSON when the model wraps it in prose. Pydantic validation then rejects any
value outside the enum or outside the bounds, rather than letting an
approximate field flow downstream.

Field names and enum values are in French, matching the interface.

Analysed emails accumulate in a table sorted by descending urgency. Five demo
emails ship with the repo, and a free-text mode accepts your own message.

---

### 3. Self-service analytics

Converts a French question into an interactive chart over the Superstore
dataset, roughly ten thousand order lines.

The question is translated into an analysis spec: grouping dimension, metric,
aggregation function, chart type, time window. That spec is then executed with
pandas and rendered with Plotly.

**This translation uses keyword matching, not a language-model call.** The
choice is deliberate: over a closed business vocabulary, an explicit rule is
deterministic, instant and free. The trade-off is that an unexpected phrasing
falls back to the defaults, sales by region. Recognised dimensions are region,
country, state, city, category, sub-category, segment and ship mode. Metrics
are sales, profit, quantity and return rate.

Column names are resolved against the columns actually present, so both
Superstore namings work (`State` or `State/Province`, `Country` or
`Country/Region`).

This is the only tab that works without an API key.

---

## Setup

Python 3.10 or later.

```bash
git clone https://github.com/Demba09/ai_portfolio
cd ai_portfolio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tabs 1 and 2 call the OpenAI API. Put a key in a `.env` file at the root:

```
OPENAI_API_KEY=sk-...
```

That file is git-ignored.

Then:

```bash
streamlit run app.py
```

On macOS, `faiss-cpu` sometimes fails to install through pip. The app detects
its absence and reports it instead of crashing. Fallback:

```bash
conda install -c conda-forge faiss-cpu
```

---

## Repository layout

```
app.py                        Streamlit app, all three tabs
requirements.txt              dependencies
data/
  inca_boamp.pdf              public tender notice, the RAG demo document
  emails_demo.jsonl           five support emails
  superstore_orders.csv       orders, about 10,000 rows
  superstore_returns.csv      returns, used for the return-rate metric
CODE_REVIEW.md                internal code review and fix tracking
```

---

## Known limitations

These are demonstrators meant to show an approach, not systems to run as-is.
The gaps, stated plainly:

- **The RAG re-indexes the document on every question.** The FAISS index is not
  cached between calls. Fine for a demo PDF, wrong for a large corpus, where
  indexing should be decoupled from querying and persisted.
- **No measured retrieval evaluation.** Answer quality was checked by hand on
  the demo document. There is no question-answer test set and no recall metric.
- **Fixed-size chunking**, blind to document structure. On heavily structured
  text, section-aware chunking would retrieve better.
- **State lives in the Streamlit session.** Nothing survives closing the tab.
- **No automated tests.**

---

## Privacy note

All data in this repository is public or synthetic. The PDF is a procurement
notice published on BOAMP, the emails are fictional, and Superstore is a widely
circulated demo dataset. No real client data and no personal information is
present.
