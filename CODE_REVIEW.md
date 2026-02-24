# Revue du Code - Portfolio IA

## 📊 Résumé Général
- **Lignes de code** : ~1017
- **Composants** : 3 tabs Streamlit (RAG, Triage Email, Analytics)
- **État global** : Bon, avec quelques bugs et optimisations possibles

---

## 🔴 Problèmes Critiques

### 1. **Duplication de logique dans `run_spec()` (lignes 560-620)**
```python
# Problème: le code crée DEUX fois les graphiques bar/line/pie
# - Première fois: lignes 560-575 (if/elif/elif)
# - Deuxième fois: lignes 580-615 (else block avec duplication)
```
**Impact** : Code dupliqué, difficile à maintenir
**Fix** : Réfactoriser en une seule section

### 2. **JSON malformé potentiel dans `triage_email_llm()` (ligne 75)**
```python
data = json.loads(raw)  # Peut échouer si LLM retourne du texte extra
```
**Impact** : Crash si le LLM ne produit pas du JSON pur
**Fix** : Ajouter extraction JSON robuste avec regex

### 3. **Pas de vérification du client OpenAI dans les fonction RAG**
```python
def embed_texts(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(...)  # ← client peut être None
```
**Impact** : Crash à l'exécution si API key manquante
**Fix** : Vérifier `client is not None` avant d'appeler

### 4. **Import redondant de `Tuple` (ligne 10)**
```python
from typing import Tuple
# Ensuite: Tuple["faiss.IndexFlatIP", ...]  ← utilisation cohérente, OK
```
Pas critique mais noté.

---

## 🟡 Problèmes Majeurs

### 1. **Duplicate Code dans `run_spec()` - LIGNE 560-615**
```python
# Vous créez les graphiques 2 fois!
# Première section (560-575): if/elif pour bar/line
# Puis else block (580-615): refait bar/line/pie

# Au lieu de ça, on devrait avoir:
if chart_type == "bar":
    fig = px.bar(...)
elif chart_type == "line":
    fig = px.line(...)
elif chart_type == "pie" and num_groups <= 6:
    fig = px.pie(...)
else:
    fig = px.bar(...)  # fallback
```

### 2. **Détection de colonne dans `llm_to_spec_fr()` - REDONDANCE**
```python
# Lignes 380-395: Détection de métrique (non-timeseries)
# Lignes 420-430: MÊME détection répétée pour timeseries+non-timeseries
```
**Impact** : Réduire duplication code

### 3. **Gestion d'erreur faible dans `load_superstore_data()`**
```python
# Pas de timeout robuste pour GitHub
# Pas de retry logic
# Pas d'indication à l'utilisateur du chargement en cours
```

### 4. **Keyword lists dans `llm_to_spec_fr()` non-optimisées**
```python
# 15+ listes de keywords créées à chaque appel
# Mieux: les définir une seule fois comme constantes en haut du fichier
```

---

## 🟢 Problèmes Mineurs / Optimisations

### 1. **Message d'erreur utilisateur (Tab 1, ligne ~760)**
```python
st.error("Aucun PDF disponible (démo introuvable ou upload manquant).")
st.stop()
```
**Bon** : Mais peut être plus spécifique

### 2. **Cache Strategy**
- ✅ Tab 1: Pas de cache (correct, PDFs varient)
- ✅ Tab 2: Cache globale (emails_demo.jsonl statique)
- ⚠️ Tab 3: `@st.cache_resource` pour `load_superstore_data()` (OK mais longue durée)

### 3. **Pas de validation des données dans `run_spec()`**
```python
# Pas de check si la colonne existe réellement dans le dataframe
# Pas de check si le groupby existe
# Peut causer KeyError à l'exécution
```

### 4. **Type hints incomplets**
```python
def run_spec(data: pd.DataFrame, spec: dict) -> Tuple:  # Tuple de quoi?
# Mieux: -> Tuple[go.Figure, str]:
```

---

## 📋 Recommandations

### Priority 1 (Urgent)
1. ✅ **Fixer la duplication dans `run_spec()`** - Réfactoriser les graphiques
2. ✅ **Vérifier client OpenAI** dans `embed_texts()`, `triage_email_llm()`, `answer_with_citations()`
3. ✅ **Robustifier JSON parsing** dans `triage_email_llm()` avec regex fallback

### Priority 2 (Important)
4. **Extraire keyword lists** en constantes globales
5. **Supprimer duplication** détection métrique en Tab 3
6. **Ajouter validation** colonnes/groupby dans `run_spec()`
7. **Meilleur type hints** (Tuple[Figure, str] etc.)

### Priority 3 (Nice-to-have)
8. **Ajouter spinner** durant chargement Superstore GitHub
9. **Ajouter retry logic** pour téléchargements GitHub
10. **Meilleur logging** structured (logger.debug vs print)
11. **Docstrings** pour chaque fonction

---

## 📊 Qualité du Code

| Aspect | Note | Commentaire |
|--------|------|------------|
| Fonctionnalité | 8/10 | Tout fonctionne, quelques edge cases |
| Maintenabilité | 6/10 | Code dupliqué, pas de constantes |
| Performance | 8/10 | Cache stratégique, bon |
| Erreurs | 5/10 | Gestion faible en certains points |
| Documentation | 4/10 | Peu de docstrings/comments |
| Type Safety | 6/10 | Type hints partiels |

**Score global : 6.2/10** (Acceptable mais à améliorer)

---

## 🎯 Prochaines Étapes

1. Fixer duplication dans `run_spec()`
2. Vérifier les clients OpenAI
3. Refactoriser keyword detection
4. Ajouter docstrings
5. Améliorer error handling
