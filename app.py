import json
from pathlib import Path
import pandas as pd
import streamlit as st
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from typing import Tuple

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Literal, Optional





Sentiment = Literal["Très en colère", "En colère", "Neutre", "Satisfait"]
Categorie = Literal["Matériel", "Logiciel", "Accès / Identité", "Réseau", "Sécurité", "Demande de service", "Autre"]

class TicketTriage(BaseModel):
    sentiment: Sentiment
    urgence: int = Field(..., ge=1, le=5)
    categorie: Categorie
    action_immediate: str = Field(..., min_length=5)

def triage_email_llm(subject: str, body: str) -> TicketTriage:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY manquante")

    instructions = (
        "Tu es un assistant de triage pour support IT N1.\n"
        "Tu dois produire UNIQUEMENT un JSON valide (sans texte autour) respectant exactement ce schéma:\n"
        "{\"sentiment\": \"Très en colère|En colère|Neutre|Satisfait\", \"urgence\": 1..5, \"categorie\": \"Matériel|Logiciel|Accès / Identité|Réseau|Sécurité|Demande de service|Autre\", \"action_immediate\": \"...\"}\n\n"
        "RÈGLES SENTIMENT (TRÈS IMPORTANT):\n"
        "- 'Très en colère': mots forts (catastrophe, inacceptable, arnaque, furieux), exclamations (!!!), menaces, insultes\n"
        "- 'En colère': frustration visible (déçu, énervé, problème récurrent, très urgent), ton agressif, exclamation (!)\n"
        "- 'Neutre': simple demande, constat factuel, pas d'émotion marquée (création compte, info, question simple)\n"
        "- 'Satisfait': remerciements, compliments, ton positif, merci\n\n"
        "Règles urgence:\n"
        "- 5 : prod down, sécurité critique (phishing/malware), VIP, 'urgent', 'immédiat', 'dans 10 minutes'\n"
        "- 4 : blocage fort utilisateur, impact multi-personnes, VPN/SSO bloquant\n"
        "- 3 : blocage utilisateur standard sans deadline immédiate\n"
        "- 2 : gêne / contournement possible\n"
        "- 1 : demande non urgente (création compte, installation logiciel planifiée)\n\n"
        "Action immédiate: une phrase courte, opérationnelle (N1), ex: 'Appeler l'utilisateur, collecter logs, escalader N2'.\n"
        "Ne mets pas de données personnelles dans la réponse."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"OBJET:\n{subject}\n\nEMAIL:\n{body}"}
        ]
    )

    raw = resp.choices[0].message.content.strip()
    
    # Robust JSON extraction (handle extra text before/after JSON)
    import re
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {raw[:200]}... Error: {e}")
    
    return TicketTriage(**data)

load_dotenv()
# Create OpenAI client only if API key is available to avoid crashing at import time
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if _OPENAI_KEY:
    try:
        client = OpenAI(api_key=_OPENAI_KEY)
    except Exception:
        client = None
else:
    client = None

# Définir le répertoire data en chemin absolu (pour Streamlit Cloud et déploiements)
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"

# ---------- RAG utils (PDF -> chunks -> FAISS -> LLM) ----------

if not FAISS_AVAILABLE:
    # Provide a clear runtime hint when user tries to use RAG features.
    # Installation recommendation (macOS / conda):
    #   conda install -c conda-forge faiss-cpu
    # Or try: pip install faiss-cpu (may fail on macOS)
    pass

def extract_pdf_text(pdf_bytes: bytes) -> list[dict]:
    """Return list of {page, text}."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        raise RuntimeError(f"Impossible d'ouvrir le PDF : {e}")

    pages = []
    try:
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = " ".join(text.split())
            if text.strip():
                pages.append({"page": i, "text": text})
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'extraction des pages : {e}")

    return pages

def chunk_pages(pages: list[dict], chunk_size: int = 1200, overlap: int = 200) -> list[dict]:
    """Return list of {chunk_id, page, text}."""
    chunks = []
    cid = 0
    for p in pages:
        t = p["text"]
        start = 0
        while start < len(t):
            end = min(len(t), start + chunk_size)
            chunk_text = t[start:end].strip()
            if chunk_text:
                chunks.append({"chunk_id": cid, "page": p["page"], "text": chunk_text})
                cid += 1
            if end == len(t):
                break
            start = max(0, end - overlap)
    return chunks

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embeddings matrix shape (n, d) float32 normalized."""
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")
    
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = np.array([e.embedding for e in resp.data], dtype=np.float32)
    # normalize for cosine similarity via inner product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def build_faiss_index(chunks: list[dict]) -> Tuple["faiss.IndexFlatIP", list[dict]]:
    vecs = embed_texts([c["text"] for c in chunks])
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index, chunks

def retrieve(index: faiss.IndexFlatIP, chunks: list[dict], question: str, top_k: int = 5) -> list[dict]:
    qv = embed_texts([question])
    scores, ids = index.search(qv, top_k)
    results = []
    for rank, idx in enumerate(ids[0].tolist()):
        if idx == -1:
            continue
        c = chunks[idx]
        results.append({
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "text": c["text"],
        })
    return results

def answer_with_citations(question: str, retrieved: list[dict]) -> Tuple[str, list[dict]]:
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[C{r['chunk_id']} | p.{r['page']}] {r['text']}"
        )
    context = "\n\n".join(context_blocks)

    instructions = (
        "Tu es un assistant d'analyse d'appel d'offres. "
        "Réponds uniquement à partir des extraits fournis. "
        "Si la réponse n'est pas dans les extraits, dis clairement : "
        "'Information non trouvée dans le document fourni.' "
        "Ne devine pas. "
        "À la fin de chaque phrase factuelle, ajoute une citation sous forme [Cxx] "
        "en utilisant les IDs des extraits."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nEXTRAITS:\n{context}"}
        ]
    )
    return resp.choices[0].message.content, retrieved

# ---------- Utils ----------
@st.cache_data
def load_emails_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

@st.cache_data
def load_superstore_xls(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)

    if "Orders" not in xls.sheet_names:
        raise ValueError(
            f"La feuille 'Orders' est introuvable. Feuilles disponibles : {xls.sheet_names}"
        )

    df = pd.read_excel(xls, sheet_name="Orders")

    # Nettoyage colonnes
    df.columns = [c.strip() for c in df.columns]

    # Normalisation dates
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"])

    return df

# ---------- Tab 3 Data Loading & LLM Utils ----------
# Load Superstore data for Tab 3
superstore_path = DATA_DIR / "superstore.xlsx"
superstore_orders_csv = DATA_DIR / "superstore_orders.csv"
superstore_returns_csv = DATA_DIR / "superstore_returns.csv"

@st.cache_resource
def load_superstore_data():
    """Load superstore data from CSV/Excel/GitHub."""
    import requests
    import io
    
    df = None
    
    # Essai 1: charger depuis CSV local (plus rapide et plus léger)
    if superstore_orders_csv.exists():
        try:
            df = pd.read_csv(superstore_orders_csv)
            df["Order Date"] = pd.to_datetime(df["Order Date"])
            df["Ship Date"] = pd.to_datetime(df["Ship Date"])
            print(f"[DEBUG] Loaded Orders from CSV: {df.shape[0]} rows")
            
            # Load returns
            if superstore_returns_csv.exists():
                returns_df = pd.read_csv(superstore_returns_csv)
                returned_order_ids = set(returns_df["Order ID"].unique())
                df["Returned"] = df["Order ID"].isin(returned_order_ids).astype(int)
                print(f"[DEBUG] Loaded {len(returned_order_ids)} returns from CSV")
            else:
                df["Returned"] = 0
            
            return df
        except Exception as e:
            print(f"[DEBUG] Error loading CSV: {e}")
    
    # Essai 2: charger depuis le fichier Excel local
    if superstore_path.exists():
        try:
            df = load_superstore_xls(superstore_path)
            print(f"[DEBUG] Loaded from local Excel: {superstore_path}")
            
            # Load returns sheet
            try:
                returns_df = pd.read_excel(superstore_path, sheet_name="Returns")
                returns_df.columns = [c.strip() for c in returns_df.columns]
                returned_order_ids = set(returns_df["Order ID"].unique())
                df["Returned"] = df["Order ID"].isin(returned_order_ids).astype(int)
                print(f"[DEBUG] Loaded {len(returned_order_ids)} returns from Excel")
            except Exception as e:
                print(f"[DEBUG] Could not load Returns sheet: {e}")
                df["Returned"] = 0
            
            return df
        except Exception as e:
            print(f"[DEBUG] Error loading Excel: {e}")
    
    # Essai 3: télécharger depuis GitHub (raw CSV)
    print(f"[DEBUG] Tentative de téléchargement depuis GitHub...")
    try:
        orders_url = "https://raw.githubusercontent.com/Demba09/ai_portfolio/main/data/superstore_orders.csv"
        returns_url = "https://raw.githubusercontent.com/Demba09/ai_portfolio/main/data/superstore_returns.csv"
        
        df = pd.read_csv(orders_url)
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        df["Ship Date"] = pd.to_datetime(df["Ship Date"])
        print(f"[DEBUG] Loaded Orders from GitHub CSV: {df.shape[0]} rows")
        
        # Load returns
        try:
            returns_df = pd.read_csv(returns_url)
            returned_order_ids = set(returns_df["Order ID"].unique())
            df["Returned"] = df["Order ID"].isin(returned_order_ids).astype(int)
            print(f"[DEBUG] Loaded {len(returned_order_ids)} returns from GitHub CSV")
        except Exception as e:
            print(f"[DEBUG] Could not load Returns from GitHub: {e}")
            df["Returned"] = 0
        
        return df
    except Exception as e:
        print(f"[DEBUG] Error loading from GitHub: {e}")
        return None

df = load_superstore_data()

if df is not None:
    print(f"[DEBUG] Successfully loaded Superstore: {df.shape[0]} rows, {df.shape[1]} cols")

def llm_to_spec_fr(question: str) -> dict:
    """Convert French question to a spec dict using extensive pattern matching."""
    question_lower = question.lower()
    
    # Default values
    spec = {
        "chart_type": "bar",
        "groupby": "Region",
        "agg": "sum",
        "column": "Sales",
        "is_timeseries": False,
        "is_return_rate": False,
        "time_range_days": None
    }
    
    # ===== RETURN RATE DETECTION =====
    return_keywords = ["retour", "return", "taux de retour", "return rate", "returned", "retourné", "taux retour", "annulation"]
    is_return_rate = any(kw in question_lower for kw in return_keywords)
    
    if is_return_rate:
        spec["is_return_rate"] = True
        # Pour le taux de retour, on utilise une aggregation spéciale (calcul du %) plutôt que une colonne
        spec["agg"] = "mean"  # Mean of 0/1 values = return rate percentage
        spec["chart_type"] = "bar"
    
    # ===== TEMPORAL QUERIES (TIMESERIES) =====
    temporal_keywords = [
        "évolution", "trend", "evolution", "jour", "jours", "mois", "semaine", "week",
        "date", "temps", "time", "over time", "derniers", "últimos", "past", "history",
        "chronologique", "chronologic", "timeline", "historique", "progression", "variation"
    ]
    is_temporal = any(kw in question_lower for kw in temporal_keywords) and not is_return_rate
    
    if is_temporal:
        spec["is_timeseries"] = True
        spec["groupby"] = "Order Date"
        spec["chart_type"] = "line"
        
        # Detect time range (30 days, 1 month, etc.)
        if "30" in question_lower or "dernier mois" in question_lower:
            spec["time_range_days"] = 30
        elif "7" in question_lower or "semaine" in question_lower or "week" in question_lower:
            spec["time_range_days"] = 7
        elif "90" in question_lower or "trimestre" in question_lower or "quarter" in question_lower:
            spec["time_range_days"] = 90
        elif "365" in question_lower or "an" in question_lower or "year" in question_lower or "année" in question_lower:
            spec["time_range_days"] = 365
        elif "mois" in question_lower or "month" in question_lower:
            spec["time_range_days"] = 30
        else:
            spec["time_range_days"] = 30  # Default 30 days
    else:
        # ===== METRIC DETECTION (NON-TIMESERIES) =====
        profit_keywords = ["profit", "rentabilité", "rentabilite", "marge", "margin", "bénéfice", "benefice", "gain", "revenus nets", "net income"]
        quantity_keywords = ["quantité", "quantity", "quantite", "nombre", "count", "volume", "qté", "nb", "commandes", "orders", "articles", "items"]
        sales_keywords = ["ventes", "sales", "chiffre affaires", "ca", "revenue", "revenus", "montant", "amount", "total"]
        
        if any(kw in question_lower for kw in profit_keywords):
            spec["column"] = "Profit"
        elif any(kw in question_lower for kw in quantity_keywords):
            spec["column"] = "Quantity"
        elif any(kw in question_lower for kw in sales_keywords):
            spec["column"] = "Sales"
        else:
            spec["column"] = "Sales"  # Default
        
        # ===== GROUPING DIMENSION DETECTION =====
        region_keywords = ["région", "region", "régions", "regions", "pays", "country", "zone", "area", "territoire", "territory"]
        category_keywords = ["catégorie", "category", "categorie", "categories", "type", "genre", "classe", "class"]
        subcategory_keywords = ["sous-catégorie", "sous catégorie", "subcategory", "sub-category", "sous-cat", "souscatégorie"]
        segment_keywords = ["segment", "segments", "clientèle", "clientele", "population", "groupe", "group", "partie"]
        state_keywords = ["état", "state", "province", "états", "states", "provinces"]
        city_keywords = ["ville", "city", "villes", "cities", "cité", "localité", "localite"]
        shipmode_keywords = ["mode expédition", "mode d'expédition", "ship mode", "shipping", "expédition", "livraison", "transport", "logistique"]
        
        if any(kw in question_lower for kw in subcategory_keywords):
            spec["groupby"] = "Sub-Category"
        elif any(kw in question_lower for kw in category_keywords):
            spec["groupby"] = "Category"
        elif any(kw in question_lower for kw in segment_keywords):
            spec["groupby"] = "Segment"
        elif any(kw in question_lower for kw in state_keywords):
            spec["groupby"] = "State"
        elif any(kw in question_lower for kw in city_keywords):
            spec["groupby"] = "City"
        elif any(kw in question_lower for kw in shipmode_keywords):
            spec["groupby"] = "Ship Mode"
        elif any(kw in question_lower for kw in region_keywords):
            spec["groupby"] = "Region"
        else:
            spec["groupby"] = "Region"  # Default
        
        # ===== CHART TYPE DETECTION (IMPROVED LOGIC) =====
        # Vérifier d'abord les mots-clés explicites du graphique
        bar_keywords = ["barre", "bar", "colonne", "column", "histogram", "histogramme", "diagramme en barres", "comparaison"]
        line_keywords = ["ligne", "line", "courbe", "curve", "graphique linéaire", "line chart"]
        pie_keywords = ["camembert", "pie", "secteurs", "sectors", "pourcentages", "percentage", "parts", "portion", "distribution"]
        
        if any(kw in question_lower for kw in pie_keywords):
            spec["chart_type"] = "pie"
        elif any(kw in question_lower for kw in line_keywords):
            spec["chart_type"] = "line"
        elif any(kw in question_lower for kw in bar_keywords):
            spec["chart_type"] = "bar"
        else:
            # Logique automatique: choisir le type en fonction du contexte
            # Distribution/parts -> pie, Comparaison -> bar
            if any(kw in question_lower for kw in ["distribution", "répartition", "parts", "portion", "pourcentage"]):
                spec["chart_type"] = "pie"
            else:
                spec["chart_type"] = "bar"  # Default pour comparaisons
    
    # ===== METRIC DETECTION (APPLIES TO BOTH TIMESERIES AND NON-TIMESERIES) =====
    if not is_return_rate:
        profit_keywords = ["profit", "rentabilité", "rentabilite", "marge", "margin", "bénéfice", "benefice", "gain", "revenus nets", "net income"]
        quantity_keywords = ["quantité", "quantity", "quantite", "nombre", "count", "volume", "qté", "nb", "commandes", "orders", "articles", "items"]
        
        if any(kw in question_lower for kw in profit_keywords):
            spec["column"] = "Profit"
        elif any(kw in question_lower for kw in quantity_keywords):
            spec["column"] = "Quantity"
        else:
            spec["column"] = "Sales"  # Default
    
    # ===== AGGREGATION DETECTION =====
    sum_keywords = ["somme", "sum", "total", "totalité", "cumulé", "cumulative", "addition"]
    mean_keywords = ["moyenne", "mean", "average", "moyen", "avg", "médiane", "median", "taux", "rate", "pourcentage", "percentage"]
    count_keywords = ["nombre", "count", "combien", "how many", "total nombre", "nb", "occurence", "occurrence"]
    min_keywords = ["minimum", "min", "plus bas", "lowest", "moins", "smallest"]
    max_keywords = ["maximum", "max", "plus haut", "highest", "plus grand", "largest"]
    
    if any(kw in question_lower for kw in max_keywords):
        spec["agg"] = "max"
    elif any(kw in question_lower for kw in min_keywords):
        spec["agg"] = "min"
    elif any(kw in question_lower for kw in count_keywords):
        spec["agg"] = "count"
    elif any(kw in question_lower for kw in mean_keywords):
        spec["agg"] = "mean"
    elif any(kw in question_lower for kw in sum_keywords):
        spec["agg"] = "sum"
    else:
        spec["agg"] = "sum"  # Default
    
    return spec

def run_spec(data: pd.DataFrame, spec: dict) -> Tuple:
    """Execute spec and return (figure, insight text) using Plotly for beautiful interactive charts."""
    if data is None:
        raise ValueError("No data available")
    
    chart_type = spec.get("chart_type", "bar")
    groupby = spec.get("groupby", "Region")
    agg = spec.get("agg", "sum")
    column = spec.get("column", "Sales")
    is_timeseries = spec.get("is_timeseries", False)
    is_return_rate = spec.get("is_return_rate", False)
    time_range_days = spec.get("time_range_days", 30)
    
    # Handle timeseries (date-based) queries
    working_data = data.copy()
    if is_timeseries and "Order Date" in working_data.columns:
        # Ensure Order Date is datetime
        working_data["Order Date"] = pd.to_datetime(working_data["Order Date"], errors="coerce")
        
        # Filter to the last N days
        max_date = working_data["Order Date"].max()
        min_date = max_date - timedelta(days=time_range_days)
        working_data = working_data[working_data["Order Date"] >= min_date]
        
        # Group by date and aggregate
        if agg == "sum":
            grouped_data = working_data.groupby("Order Date")[column].sum().reset_index()
        elif agg == "mean":
            grouped_data = working_data.groupby("Order Date")[column].mean().reset_index()
        elif agg == "count":
            grouped_data = working_data.groupby("Order Date")[column].count().reset_index(name=column)
        elif agg == "max":
            grouped_data = working_data.groupby("Order Date")[column].max().reset_index()
        elif agg == "min":
            grouped_data = working_data.groupby("Order Date")[column].min().reset_index()
        else:
            grouped_data = working_data.groupby("Order Date")[column].sum().reset_index()
        
        # Sort by date
        grouped_data = grouped_data.sort_values("Order Date")
        
        # Create line chart for timeseries
        fig = px.line(
            grouped_data,
            x="Order Date",
            y=column,
            title=f"Évolution de {column} (derniers {time_range_days} jours) - {agg.upper()}",
            markers=True,
            height=500,
            labels={column: f"{column} ({agg})", "Order Date": "Date"}
        )
        
        insight = f"Évolution de {column} sur {time_range_days} jours ({agg.upper()}). Total: {grouped_data[column].sum():.2f}. Moyenne: {grouped_data[column].mean():.2f}. Min: {grouped_data[column].min():.2f}. Max: {grouped_data[column].max():.2f}."
    elif is_return_rate and "Returned" in working_data.columns:
        # Calculate return rate by groupby dimension
        return_rate_data = working_data.groupby(groupby).agg({
            "Returned": ["sum", "count"]
        }).reset_index()
        
        return_rate_data.columns = [groupby, "Returned_Count", "Total_Orders"]
        return_rate_data["Return_Rate_Pct"] = (return_rate_data["Returned_Count"] / return_rate_data["Total_Orders"] * 100).round(2)
        
        # Create bar chart for return rates
        fig = px.bar(
            return_rate_data,
            x=groupby,
            y="Return_Rate_Pct",
            title=f"Taux de retour par {groupby}",
            color="Return_Rate_Pct",
            color_continuous_scale="RdYlGn_r",  # Red for high return rates
            height=500,
            labels={"Return_Rate_Pct": "Taux de retour (%)", groupby: groupby}
        )
        
        # Add percentage text on bars
        fig.update_traces(text=return_rate_data["Return_Rate_Pct"].apply(lambda x: f"{x}%"), textposition="outside")
        
        insight = f"Taux moyen de retour: {return_rate_data['Return_Rate_Pct'].mean():.2f}%. Pire: {return_rate_data.loc[return_rate_data['Return_Rate_Pct'].idxmax(), groupby]} ({return_rate_data['Return_Rate_Pct'].max():.2f}%). Meilleur: {return_rate_data.loc[return_rate_data['Return_Rate_Pct'].idxmin(), groupby]} ({return_rate_data['Return_Rate_Pct'].min():.2f}%)."
    else:
        # Aggregate data by groupby dimension
        if agg == "sum":
            grouped_data = working_data.groupby(groupby)[column].sum().reset_index()
        elif agg == "mean":
            grouped_data = working_data.groupby(groupby)[column].mean().reset_index()
        elif agg == "count":
            grouped_data = working_data.groupby(groupby).size().reset_index(name=column)
        elif agg == "max":
            grouped_data = working_data.groupby(groupby)[column].max().reset_index()
        elif agg == "min":
            grouped_data = working_data.groupby(groupby)[column].min().reset_index()
        else:
            grouped_data = working_data.groupby(groupby)[column].sum().reset_index()
        
        # Create chart based on type and number of groups
        # Note: Pie charts sont lisibles seulement avec peu de groupes (<7)
        num_groups = len(grouped_data)
        
        if chart_type == "pie" and num_groups <= 6:
            fig = px.pie(
                grouped_data,
                names=groupby,
                values=column,
                title=f"Distribution de {column} par {groupby}",
                height=500
            )
        elif chart_type == "line":
            fig = px.line(
                grouped_data,
                x=groupby,
                y=column,
                title=f"{agg.upper()} de {column} par {groupby}",
                markers=True,
                height=500,
                labels={column: f"{column} ({agg})", groupby: groupby}
            )
        else:
            # Bar chart (par défaut ou si pie chart a trop d'items)
            fig = px.bar(
                grouped_data,
                x=groupby,
                y=column,
                title=f"{agg.upper()} de {column} par {groupby}",
                color=column,
                color_continuous_scale="Blues",
                height=500,
                labels={column: f"{column} ({agg})", groupby: groupby}
            )
        
        # Generate insight
        total = grouped_data[column].sum()
        max_group = grouped_data.loc[grouped_data[column].idxmax()]
        min_group = grouped_data.loc[grouped_data[column].idxmin()]
        insight = f"Total: {total:.2f}. Meilleur {groupby}: '{max_group[groupby]}' ({max_group[column]:.2f}). Pire: '{min_group[groupby]}' ({min_group[column]:.2f})."
    
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        hovermode="x unified"
    )
    
    return fig, insight

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Portfolio IA — Nexton", layout="wide")
st.title("Portfolio IA — 3 mini-projets")

tab1, tab2, tab3 = st.tabs([
    "1) Analyse documentaire stratégique (RAG)",
    "2) Automatisation du support (LLM + Extraction structurée)",
    "3) Self-Service Analytics (NL → Insights)"
])


# ---------- Tab 1 ----------
with tab1:
    st.subheader("Assistant Appels d'Offres (PDF)")
    st.caption("Démo : PDF d'un appel d'offres ou import d'un PDF → Q/R factuelle + citations.")

    pdf_path = DATA_DIR / "inca_boamp.pdf"

    # init état
    if "p1_use_demo" not in st.session_state:
        st.session_state["p1_use_demo"] = True  # par défaut: démo activée

    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("### Document")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("✅ Utiliser le PDF de démo", use_container_width=True):
                st.session_state["p1_use_demo"] = True
                if not st.session_state.get("p1_question"):
                    st.session_state["p1_question"] = "Quelles sont les attentes en data science / IA ?"

        with c2:
            if st.button("📄 Importer mon PDF", use_container_width=True):
                st.session_state["p1_use_demo"] = False

        use_demo = st.session_state["p1_use_demo"]

        pdf_file = None
        if use_demo:
            if pdf_path.exists():
                st.success("Mode démo activé (Appel d'offre gestion de données).")
            else:
                st.warning("PDF de démo introuvable : place `inca_boamp.pdf` dans `data/`.")
        else:
            st.info("Mode import : charge un PDF.")
            pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")

        st.markdown("### Questions de démo")
        if st.button("Comment est structurée la réponse à l'appel d'offres ?"):
            st.session_state["p1_question"] = "Comment est structurée la réponse à l'appel d'offres ? (ex: lots, étapes, critères)"
        if st.button("Qui a lancé l'appel d'offres ?"):
            st.session_state["p1_question"] = "Qui a lancé l'appel d'offres"
        if st.button("Qu'est ce qui est demandé dans l'appel d'offres ?"):
            st.session_state["p1_question"] = "Qu'est ce que que l'organisme acheteur attend des prestataires ?"

    with colB:
        st.markdown("### Question")
        question = st.text_input(
            "Pose une question",
            key="p1_question",
            placeholder="Ex: Quelles sont les attentes en data science / IA ?"
        )

        # Choix du PDF : démo ou upload
        pdf_bytes = None
        pdf_label = None

        if st.session_state["p1_use_demo"]:
            if pdf_path.exists():
                pdf_bytes = pdf_path.read_bytes()
                pdf_label = "PDF de démo (inca_boamp.pdf)"
        else:
            if pdf_file is not None:
                pdf_bytes = pdf_file.getvalue()
                pdf_label = "PDF uploadé"

        if pdf_label:
            st.caption(f"Source : {pdf_label}")

        if st.button("Répondre", type="primary"):
            if client is None:
                st.error("OPENAI_API_KEY manquante : configure-la dans .env (ou export terminal) puis relance.")
                st.stop()
            if not FAISS_AVAILABLE:
                st.error("FAISS non disponible. Installe faiss-cpu (ou on passe en fallback sans FAISS).")
                st.stop()
            if pdf_bytes is None:
                st.error("Aucun PDF disponible (démo introuvable ou upload manquant).")
                st.stop()
            if not question.strip():
                st.error("Merci de saisir une question.")
                st.stop()

            with st.spinner("Indexation + recherche + réponse..."):
                try:
                    pages = extract_pdf_text(pdf_bytes)
                except Exception as e:
                    st.error(f"Erreur lors de l'extraction du PDF : {e}")
                    st.stop()
                if not pages:
                    st.error("Impossible d'extraire du texte du PDF (scan/protégé).")
                    st.stop()

                chunks = chunk_pages(pages)
                index, chunks = build_faiss_index(chunks)
                hits = retrieve(index, chunks, question, top_k=5)
                answer, used = answer_with_citations(question, hits)

            st.markdown("### Réponse")
            st.write(answer)

            st.markdown("### Extraits utilisés (sources)")
            for r in used:
                st.markdown(f"**[C{r['chunk_id']}]** (page {r['page']})")
                st.caption(r["text"])
# ---------- Tab 2 ----------
with tab2:
    st.subheader("Trieur Support IT (LLM + Extraction structurée)")
    st.caption("Sélection d’un email → extraction structurée (JSON) → tableau triable.")

    emails_path = DATA_DIR / "emails_demo.jsonl"
    if not emails_path.exists():
        st.error("emails_demo.jsonl introuvable dans `data/`.")
        st.stop()

    df_emails = load_emails_jsonl(emails_path)

    # état: historique des analyses
    if "p2_results" not in st.session_state:
        st.session_state["p2_results"] = []

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("### 📧 Emails")
        
        # Mode: démo ou rédiger
        email_mode = st.radio(
            "Mode",
            options=["📋 Emails de démo", "✏️ Rédiger mon email"],
            horizontal=True,
            key="p2_email_mode"
        )
        
        if email_mode == "📋 Emails de démo":
            st.dataframe(df_emails[["id", "subject", "from", "timestamp"]], use_container_width=True, hide_index=True)
            
            pick = st.selectbox("Ouvrir un email", df_emails["id"].tolist(), key="p2_pick")
            row = df_emails[df_emails["id"] == pick].iloc[0]
            st.text_input("Objet", value=row["subject"], disabled=True)
            st.text_area("Contenu", value=row["body"], height=240, disabled=True)
            
            subject = row["subject"]
            body = row["body"]
            email_source = f"Email démo: {pick}"
        else:
            subject = st.text_input(
                "Objet du mail",
                placeholder="Ex: Mon imprimante ne marche plus",
                key="p2_custom_subject"
            )
            body = st.text_area(
                "Contenu du mail",
                placeholder="Décris ton problème ici...",
                height=240,
                key="p2_custom_body"
            )
            email_source = "Email personnalisé"
        
        analyze = st.button("Analyser", type="primary", use_container_width=True)

    with colR:
        st.markdown("### Résultat structuré (JSON)")
        if analyze:
            if not subject.strip() or not body.strip():
                st.error("Merci de remplir l'objet et le contenu du mail.")
                st.stop()
            
            if client is None:
                st.error("OPENAI_API_KEY manquante : configure-la dans .env (ou export terminal) puis relance.")
                st.stop()

            with st.spinner("Analyse en cours..."):
                try:
                    triage = triage_email_llm(subject=subject, body=body)
                except Exception as e:
                    st.error(f"Erreur LLM/JSON : {e}")
                    st.stop()

            st.json(triage.model_dump())

            st.session_state["p2_results"].append({
                "source": email_source,
                "subject": subject,
                "sentiment": triage.sentiment,
                "urgence": triage.urgence,
                "categorie": triage.categorie,
                "action_immediate": triage.action_immediate,
            })

        else:
            st.info("Clique sur **Analyser** pour obtenir le JSON structuré.")

        st.markdown("### Tableau de triage")
        if st.session_state["p2_results"]:
            df_out = pd.DataFrame(st.session_state["p2_results"])
            # tri par urgence desc
            df_out = df_out.sort_values(by="urgence", ascending=False)
            st.dataframe(df_out, use_container_width=True, hide_index=True)
        else:
            st.caption("Aucun email analysé pour le moment.")

# ---------- Tab 3 ----------
with tab3:
    st.markdown("### 📊 Analyseur de Données Superstore")
    
    # Vérifier que les données sont disponibles
    if df is None:
        st.error("❌ Données Superstore non disponibles")
        st.info("Pour utiliser cette tab, place le fichier `superstore.xlsx` dans le dossier `data/` de ton projet.")
        st.stop()
    
    st.markdown("#### ✍️ Posez votre question")
    
    question = st.text_input(
        "Entrez une question en français",
        key="p3_question",
        placeholder="Ex: Affiche-moi les ventes par région, Profit moyen par catégorie, Taux de retour par segment..."
    )

    st.divider()
    
    # Préférences de graphique
    st.markdown("#### 📈 Préférences de graphique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type_choice = st.radio(
            "Type de graphique préféré :",
            options=["Automatique", "Barre", "Ligne", "Camembert"],
            horizontal=True,
            help="Sélectionnez le type de graphique que vous préférez. L'option 'Automatique' laisse le système décider selon la question."
        )
    
    with col2:
        st.info(
            "💡 **Automatique** : Le système choisit le meilleur type selon votre question.\n"
            "📊 **Barre** : Comparaison entre catégories (défaut).\n"
            "📈 **Ligne** : Évolutions temporelles ou progressions.\n"
            "🥧 **Camembert** : Distribution en pourcentages."
        )
    
    st.divider()
    
    # Panel de questions d'exemple (collapsible)
    with st.expander("💡 Voir les questions suggérées (cliquez pour utiliser)", expanded=False):
        st.markdown("Cliquez sur une question pour la charger dans la barre de recherche:")
        
        # Dictionnaire de questions organisées par catégorie
        # Les questions sont optimisées pour être bien comprises par le LLM
        question_panels = {
            "📈 Ventes par Région": [
                "Ventes par région",
                "Total des ventes par région",
                "Graphique des ventes par région en barre",
                "Distribution des ventes par région",
                "Comparaison des ventes entre régions"
            ],
            "💰 Profit par Catégorie": [
                "Profit par catégorie",
                "Profit moyen par catégorie",
                "Quel profit dans chaque catégorie",
                "Graphique du profit par catégorie",
                "Distribution des profits par catégorie"
            ],
            "📦 Quantités": [
                "Quantité totale par région",
                "Nombre de commandes par catégorie",
                "Quantité moyenne par segment",
                "Total des articles par mode expédition",
                "Combien de commandes par région"
            ],
            "❌ Retours": [
                "Taux de retour par région",
                "Retours par catégorie",
                "Taux de retour par segment",
                "Pourcentage de retours par région",
                "Moyenne des retours par state"
            ],
            "📊 Évolutions Temporelles": [
                "Évolution des ventes sur 30 jours",
                "Évolution du profit sur 7 jours",
                "Progression des ventes sur 90 jours",
                "Évolution des commandes sur 1 mois",
                "Tendance des ventes sur derniers jours"
            ],
            "📍 Par Dimension": [
                "Ventes par catégorie",
                "Profit par segment",
                "Quantité par état",
                "Ventes par mode expédition",
                "Retours par sous-catégorie"
            ],
            "📈 Analyses Avancées": [
                "Profit maximum par région",
                "Profit minimum par catégorie",
                "Moyenne des ventes par segment",
                "Somme des quantités par state",
                "Nombre moyen de commandes par région"
            ],
            "🔍 Comparaisons": [
                "Ventes par région en camembert",
                "Distribution du profit par catégorie",
                "Ligne d'évolution des ventes",
                "Graphique linéaire du profit par segment",
                "Diagramme en barre des retours par région"
            ]
        }
        
        # Créer les colonnes pour les catégories de questions
        cols = st.columns(3)
        col_idx = 0
        
        for category, questions in question_panels.items():
            col = cols[col_idx % 3]
            with col:
                st.markdown(f"**{category}**")
                for i, q in enumerate(questions):
                    def make_callback(q_text):
                        def callback():
                            st.session_state.p3_question = q_text
                        return callback
                    
                    st.button(
                        q,
                        key=f"btn_{category}_{i}",
                        on_click=make_callback(q),
                        use_container_width=True,
                        help=f"Cliquez pour utiliser: {q}"
                    )
            col_idx += 1

    st.divider()
    
    if st.button("🚀 Générer le graphique", type="primary", use_container_width=True):
        if not question.strip():
            st.error("Merci de saisir une question.")
            st.stop()
        if df is None:
            st.error(f"❌ Données Superstore non disponibles.\n\n"
                     f"Place `superstore.xlsx` dans le dossier `{DATA_DIR.absolute()}/`.\n\n"
                     f"Fichier attendu : `{(DATA_DIR / 'superstore.xlsx').absolute()}`")
            st.stop()

        with st.spinner("⏳ Interprétation + calcul + graphique..."):
            try:
                spec = llm_to_spec_fr(question)
                
                # Appliquer le choix du graphique de l'utilisateur
                if chart_type_choice == "Barre":
                    spec["chart_type"] = "bar"
                elif chart_type_choice == "Ligne":
                    spec["chart_type"] = "line"
                elif chart_type_choice == "Camembert":
                    spec["chart_type"] = "pie"
                # "Automatique" ne change rien au spec
                
            except Exception as e:
                st.error(f"Impossible d'interpréter la question : {e}")
                st.stop()

            try:
                fig, insight = run_spec(df, spec)
            except Exception as e:
                st.error(f"Erreur lors du calcul/graphique : {e}")
                st.stop()

        st.markdown("### 📊 Résultat")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📝 Analyse")
        st.info(insight)

        with st.expander("🔧 Spec interprétée (debug)", expanded=False):
            st.json(spec)