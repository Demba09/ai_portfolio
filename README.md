# Portfolio IA

Trois démonstrateurs d'IA appliquée, réunis dans une seule application Streamlit.
Chacun répond à un besoin métier concret : interroger un document contractuel,
trier des demandes entrantes, explorer des données sans écrire de requête.

L'application se lance en une commande et fonctionne sur des données réelles
fournies dans le dépôt.

```bash
streamlit run app.py
```

---

## Les trois démos

| Onglet | Problème traité | Cœur technique |
|---|---|---|
| **1. Analyse documentaire (RAG)** | Retrouver une information précise dans un appel d'offres de plusieurs pages | Embeddings + FAISS + GPT-4o, réponses sourcées |
| **2. Automatisation du support** | Qualifier un email entrant sans lecture humaine | GPT-4o + schéma Pydantic validé |
| **3. Self-service analytics** | Obtenir un graphique à partir d'une question en français | Règles de correspondance + pandas + Plotly |

---

### 1. Analyse documentaire stratégique (RAG)

Pose une question en langage naturel sur un PDF et obtient une réponse **citée**,
avec les extraits sources et leur numéro de page.

Le document de démonstration est un avis de marché public réel publié au BOAMP,
portant sur une prestation de gestion de données. Tout autre PDF peut être
chargé à la place.

**Chaîne de traitement**

1. Extraction du texte page par page avec `pypdf`
2. Découpage en segments de 1200 caractères avec un recouvrement de 200
3. Vectorisation via `text-embedding-3-small`, vecteurs normalisés
4. Indexation dans FAISS en produit scalaire, équivalent à une similarité cosinus
5. Récupération des cinq segments les plus proches de la question
6. Génération de la réponse par GPT-4o, contrainte aux seuls extraits fournis

**Le point important : le modèle a le droit de ne pas répondre.**
L'instruction système lui impose de répondre uniquement à partir des extraits
récupérés, et de retourner explicitement « Information non trouvée dans le
document fourni » lorsque la réponse n'y figure pas. Chaque affirmation
factuelle porte une citation de la forme `[Cxx]`, rattachée à une page.
Sur un document contractuel ou réglementaire, une réponse inventée coûte
plus cher qu'une absence de réponse.

Les extraits utilisés sont affichés sous la réponse, ce qui permet de
vérifier chaque affirmation à la source.

---

### 2. Automatisation du support (extraction structurée)

Transforme un email de support en enregistrement exploitable, prêt à être
routé ou priorisé.

Le modèle renvoie un JSON validé par un schéma Pydantic :

| Champ | Type | Contrainte |
|---|---|---|
| `sentiment` | énuméré | Très en colère, En colère, Neutre, Satisfait |
| `urgence` | entier | 1 à 5 |
| `categorie` | énuméré | Matériel, Logiciel, Accès / Identité, Réseau, Sécurité, Demande de service, Autre |
| `action_immediate` | texte | non vide |

Deux garde-fous encadrent la sortie du modèle. Une extraction par expression
régulière isole le JSON lorsque le modèle l'entoure de texte. La validation
Pydantic rejette ensuite toute valeur hors énumération ou hors bornes, plutôt
que de laisser passer un champ approximatif en aval.

Les emails analysés s'accumulent dans un tableau trié par urgence
décroissante. Cinq emails de démonstration sont fournis, et un mode libre
permet de saisir son propre message.

---

### 3. Self-service analytics

Convertit une question en français en un graphique interactif sur le jeu de
données Superstore, environ dix mille lignes de commandes.

La question est traduite en spécification d'analyse : dimension de
regroupement, métrique, fonction d'agrégation, type de graphique, fenêtre
temporelle. Cette spécification est ensuite exécutée avec pandas et rendue
avec Plotly.

**Cette traduction repose sur des règles de correspondance de mots-clés, pas
sur un appel à un modèle de langage.** Le choix est assumé : sur un vocabulaire
métier fermé, une règle explicite est déterministe, instantanée et gratuite.
La contrepartie est qu'une formulation inattendue retombe sur les valeurs par
défaut, ventes par région. Les dimensions reconnues sont la région, l'état, la
ville, la catégorie, la sous-catégorie, le segment et le mode d'expédition.
Les métriques sont les ventes, le profit, la quantité et le taux de retour.

Cet onglet est le seul à fonctionner sans clé API.

---

## Installation

Python 3.10 ou supérieur.

```bash
git clone https://github.com/Demba09/ai_portfolio
cd ai_portfolio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Les onglets 1 et 2 appellent l'API OpenAI. Renseignez une clé dans un fichier
`.env` à la racine :

```
OPENAI_API_KEY=sk-...
```

Ce fichier est exclu du suivi Git.

Puis :

```bash
streamlit run app.py
```

Sur macOS, `faiss-cpu` s'installe parfois mal via pip. L'application détecte
son absence et signale le problème au lieu de planter. Solution de repli :

```bash
conda install -c conda-forge faiss-cpu
```

---

## Structure du dépôt

```
app.py                        application Streamlit, les trois onglets
requirements.txt              dépendances
data/
  inca_boamp.pdf              avis de marché public, document de démo du RAG
  emails_demo.jsonl           cinq emails de support
  superstore_orders.csv       commandes, environ 10 000 lignes
  superstore_returns.csv      retours, utilisés pour le taux de retour
CODE_REVIEW.md                revue de code interne et suivi des correctifs
```

---

## Limites connues

Ces démonstrateurs servent à montrer une approche, pas à être exploités en
l'état. Les écarts assumés :

- **Le RAG réindexe le document à chaque question.** L'index FAISS n'est pas
  mis en cache entre deux appels. C'est acceptable sur un PDF de démonstration,
  pas sur un corpus volumineux, où l'indexation devrait être découplée de
  l'interrogation et persistée.
- **Pas d'évaluation chiffrée de la récupération.** La qualité des réponses a
  été vérifiée manuellement sur le document de démonstration. Aucun jeu de test
  question-réponse ni mesure de rappel n'accompagne le projet.
- **Le découpage est fait à taille fixe**, sans tenir compte de la structure du
  document. Sur des textes très structurés, un découpage guidé par les sections
  donnerait de meilleurs résultats.
- **L'état est stocké en session Streamlit.** Rien ne persiste après la
  fermeture de l'onglet du navigateur.
- **Aucun test automatisé.**

---

## Note sur la confidentialité

Toutes les données du dépôt sont publiques ou synthétiques. Le PDF provient
d'un avis de marché publié au BOAMP, les emails sont fictifs, et Superstore
est un jeu de données de démonstration largement diffusé. Aucune donnée
client ni information personnelle réelle n'est présente.
