# Zepto Capstone Project — Unified Data & AI Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-RAG%20Workflow-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-fc4c02.svg)](https://www.trychroma.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)

---

## 1. Executive Summary & Architecture

The **Zepto Capstone Project** implements an end-to-end, enterprise-grade **Data & AI Platform** for **Zepto**, India's leading 10-minute quick-commerce enterprise. The platform unites three interconnected operational pillars:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           ZEPTO UNIFIED ECOSYSTEM                         │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
┌──────────────┐             ┌──────────────┐             ┌────────────────┐
│   MODULE 1   │             │   MODULE 2   │             │    MODULE 3    │
│  Competitive │             │  Predictive  │             │   Intelligent  │
│     Data     │ ──ETL/SQL─► │  Analytics   │ ──Inference►│ Customer Care  │
│   Pipeline   │             │    & ML      │             │  RAG Assistant │
└──────────────┘             └──────────────┘             └────────────────┘
 [Web Scraping,               [EDA, Imputation,             [LangGraph Flow,
  105.50 FX INR,               Stratified Split,             ChromaDB Vector,
  SQLite Relational,           GridSearchCV,                 FastAPI /ask,
  Join Verification]           Heteroscedasticity]           Docker Container]
```

---

## 2. Repository Structure

```text
zepto-capstone/
│
├── data_pipeline/                      # Module 1: Data Engineering & ETL
│   ├── scraper.py                      # Scrapes books.toscrape.com (>=60 books, 4 categories)
│   ├── transform.py                    # Star rating mapping, availability boolean, GBP to INR (105.50)
│   ├── database.py                     # Normalized SQLite database (categories & books PK/FK)
│   ├── queries.sql                     # 5 SQL analytical queries (WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN, JOIN)
│   ├── verify_join.py                  # JOIN verification: pd.read_sql() vs pd.merge()
│   ├── run_pipeline.py                 # End-to-end master runner for Module 1
│   └── zepto.db                        # SQLite relational database
│
├── analytics/                          # Module 2: Data Science & Machine Learning
│   ├── data/
│   │   ├── ingest.py                   # Ingests once from sns.load_dataset('titanic') -> local CSV
│   │   └── titanic.csv                 # Local cached dataset (all steps read from this file)
│   ├── src/
│   │   ├── eda.py                      # Missing value policy, IQR outliers, correlation heatmap, 4-chart story
│   │   ├── classification.py           # Stratified split, ColumnTransformer pipeline, Logistic Reg, Decision Tree, RF
│   │   ├── optimization.py             # Class Imbalance (Baseline, Weights, SMOTE), GridSearchCV with OOB scoring
│   │   ├── regression.py               # Multivariate Linear Regression predicting Fare + Breusch-Pagan heteroscedasticity
│   │   └── run_analytics.py            # Master analytics runner and model exporter
│   ├── models/
│   │   └── best_titanic_pipeline.joblib# Serialized production pipeline ready for inference
│   ├── reports/
│   │   ├── figures/                    # Generated high-resolution diagnostic charts
│   │   │   ├── correlation_heatmap.png
│   │   │   ├── multivariate_data_story.png
│   │   │   ├── decision_tree.png
│   │   │   └── regression_heteroscedasticity.png
│   │   └── model_summary.csv           # Model benchmark summary metrics
│   └── requirements.txt
│
├── support_assistant/                  # Module 3: AI / RAG Deployment
│   ├── policies/                       # 8 distinct Zepto policy text files
│   │   ├── policy_1.txt                # Returns & Instant Refunds
│   │   ├── policy_2.txt                # Delivery 10-min SLA & Delays
│   │   ├── policy_3.txt                # Order Cancellation & Modifications
│   │   ├── policy_4.txt                # Zepto Pass VIP Subscription
│   │   ├── policy_5.txt                # Transparent Pricing, Surge, & Payments
│   │   ├── policy_6.txt                # Rider Safety & Anti-Harassment
│   │   ├── policy_7.txt                # Freshness & Quality Assurance
│   │   └── policy_8.txt                # Customer Privacy & Number Masking
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                   # Environment & ChromaDB settings
│   │   ├── schemas.py                  # Pydantic schemas (AskRequest, AskResponse, AgentState)
│   │   ├── retrieval.py                # ChromaDB vector store + all-MiniLM-L6-v2 embeddings
│   │   ├── graph.py                    # LangGraph StateGraph (classify_intent -> handle_general / handle_policy)
│   │   └── main.py                     # FastAPI server with POST /ask and GET /health
│   ├── chroma_db/                      # Local persistent ChromaDB vector store
│   ├── Dockerfile                      # Production Docker container configuration
│   ├── test_assistant.py               # Automated test suite (FastAPI TestClient + LangGraph)
│   └── requirements.txt
│
└── README.md                           # Master Capstone documentation
```

---

## 3. Module 1 — Data Pipeline

### 3.1 Architecture & Workflow
The data pipeline simulates continuous competitor pricing intelligence for Zepto:
1. **Extract (`scraper.py`)**: Uses `requests` and `BeautifulSoup4` to scrape live pricing, stock availability, star ratings, and titles across 4 distinct categories (*Mystery*, *Sequential Art*, *Historical Fiction*, *Travel*) from `books.toscrape.com` ($\ge 60$ books).
2. **Transform (`transform.py`)**:
   - **Star Ratings**: Word-to-integer conversion (`{'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}`).
   - **Availability**: Extracted stock text converted to strict Boolean (`True` for in-stock, `False` for out-of-stock).
   - **Currency Conversion**: GBP converted to INR with fixed rate $1\text{ GBP} = 105.50\text{ INR}$ (`price_inr`).
3. **Load (`database.py`)**: Populates a normalized relational SQLite database (`zepto.db`) with Foreign Key constraints enabled:
   ```sql
   CREATE TABLE categories (
       category_id INTEGER PRIMARY KEY AUTOINCREMENT,
       category_name TEXT UNIQUE NOT NULL
   );

   CREATE TABLE books (
       book_id INTEGER PRIMARY KEY AUTOINCREMENT,
       title TEXT NOT NULL,
       price_gbp REAL NOT NULL,
       price_inr REAL NOT NULL,
       star_rating INTEGER NOT NULL,
       availability BOOLEAN NOT NULL,
       category_id INTEGER NOT NULL,
       FOREIGN KEY (category_id) REFERENCES categories (category_id) ON DELETE CASCADE
   );
   ```

### 3.2 Analytical SQL Queries (`queries.sql`)
Demonstrates all 7 required SQL operations across 5 curated business queries:
1. **Query 1 (`WHERE`, `ORDER BY`, `LIMIT`, `JOIN`)**: Top 5 most premium in-stock titles.
2. **Query 2 (`DISTINCT`, `ORDER BY`)**: Star rating distribution and average INR pricing.
3. **Query 3 (`IN`, `JOIN`, `WHERE`, `ORDER BY`)**: Filtering books in targeted categories (`'Mystery'`, `'Travel'`) with rating $\ge 3$.
4. **Query 4 (`BETWEEN`, `WHERE`, `ORDER BY`)**: Mid-tier high-rated books priced between 2000 and 4500 INR with $\ge 4$ stars.
5. **Query 5 (`JOIN`, `GROUP BY`, `ORDER BY`)**: Category-level inventory count and pricing metrics.

### 3.3 JOIN Verification (`verify_join.py`)
Proves mathematical equivalence between Database-side SQL JOIN (`pd.read_sql`) and Client-side DataFrame JOIN (`pd.merge`) using `pandas.testing.assert_frame_equal()`.

---

## 4. Module 2 — Analytics & Machine Learning

### 4.1 Data Ingestion Safety
- Raw Titanic dataset ingested **exactly once** using `sns.load_dataset('titanic')` and immediately persisted to `analytics/data/titanic.csv`.
- All subsequent steps strictly read from the local CSV.

### 4.2 Exploratory Data Analysis (EDA)
- **Missing Value Policy**:
  - **$< 5\%$ Missing** (e.g. `embarked`, `embark_town`): Rows dropped.
  - **$5\% - 30\%$ Missing** (e.g. `age`): Imputed with median.
  - **$> 30\%$ Missing** (e.g. `deck` with ~77% missing): Dropped / flagged.
- **Univariate Analysis & Outlier Detection (IQR)**:
  $$\text{IQR} = Q_3 - Q_1,\quad \text{Lower} = Q_1 - 1.5 \times \text{IQR},\quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$
- **Bivariate Analysis**: Pearson correlation matrix visualized in `correlation_heatmap.png`.
- **Multivariate Analysis**: 4-chart survival data story in `multivariate_data_story.png`:
  1. *Survival Rate by Class & Sex* (female survival $>85\%$ in 1st/2nd class).
  2. *Age Distribution by Survival across Classes* (child survival prioritization).
  3. *Fare vs Age by Survival & Embarkation* (higher fare correlated with survival).
  4. *Family Size vs Survival* (optimal survival in 2–4 member families).

### 4.3 Predictive Modeling & Scikit-Learn Pipeline
- **Stratified Train/Test Split**: 80/20 split preserving class balance ($y = \text{survived}$).
- **Leakage Prevention**: `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` fitted **strictly on training data**.
- **Evaluated Classifiers**:
  1. *Logistic Regression*
  2. *Decision Tree* (visualized and exported to `decision_tree.png`)
  3. *Random Forest Classifier*

### 4.4 Optimization & Hyperparameter Tuning
- **Class Imbalance Comparison**: Evaluated Baseline vs. Class Weights (`class_weight='balanced'`) vs. SMOTE (`imblearn.over_sampling.SMOTE`).
- **GridSearchCV Tuning**: Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` on Random Forest.
- **Out-of-Bag (OOB) Scoring**: Validated out-of-bag generalization score ($\approx 0.82$).

### 4.5 Regression & Heteroscedasticity Analysis
- Multivariate Linear Regression predicting `fare` using `pclass`, `age`, `sibsp`, `parch`, `sex`, and `embarked`.
- **Heteroscedasticity Test**: Breusch-Pagan Lagrange Multiplier test ($p < 0.001$) and Residual vs. Fitted plot (`regression_heteroscedasticity.png`) confirming non-constant error variance in ticket pricing.

### 4.6 Model Serialization
- Best-performing pipeline serialized to `models/best_titanic_pipeline.joblib`.

---

## 5. Module 3 — Support Assistant (LangGraph RAG & FastAPI)

### 5.1 Architecture & Workflow

```
Customer Query ──► POST /ask
                         │
                         ▼
             ┌───────────────────────┐
             │ classify_intent Node  │
             └───────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
 [general_question]              [policy_question]
         │                               │
         ▼                               ▼
┌──────────────────┐            ┌──────────────────┐
│  handle_general  │            │  handle_policy   │
│ (Canned Refusal) │            │ (ChromaDB Top 3) │
└──────────────────┘            └──────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
                 Strict JSON Schema
            {"answer", "sources", "confidence"}
```

### 5.2 Zepto Policy Corpus (`policies/`)
Includes 8 distinct, comprehensive policy documents:
- `policy_1.txt`: Returns, Refund SLAs (UPI 2-4 hrs, Wallet 2 min), Perishable vs Packaged items.
- `policy_2.txt`: 10-Minute SLA, Dark Store network, weather/surge ETA adjustments, INR 50 delay compensation.
- `policy_3.txt`: Cancellation before packing (free) vs after packing (INR 30 fee) vs out for delivery.
- `policy_4.txt`: Zepto Pass membership (INR 99/mo), unlimited free deliveries, zero surge fee.
- `policy_5.txt`: MRP guarantee, surge fees (INR 10-30), UPI/Cards/COD payment methods.
- `policy_6.txt`: Rider safety, zero speed penalties, max 40 km/h speed limits, 100% direct tipping.
- `policy_7.txt`: Farm-to-dark-store quality inspection, expiry shelf life, cold-chain gel packaging.
- `policy_8.txt`: Geolocation privacy, rider virtual phone number masking, customer data deletion rights.

### 5.3 Embeddings & ChromaDB
- Local vector database in `support_assistant/chroma_db/`.
- Embedded using `sentence-transformers/all-MiniLM-L6-v2`.
- `retrieve_top_k(query, k=3)` returns top 3 matching chunks with similarity scoring.

### 5.4 LangGraph Dual-Mode Engine
- **`MOCK_LLM=1` (Deterministic Baseline)**:
  - Keyword heuristic intent classifier.
  - Returns structured templated response with policy citations and similarity confidence.
  - Zero external API dependencies.
- **`MOCK_LLM=0` (Gemini Real-LLM Path)**:
  - Intent classification via `gemini-2.5-flash`.
  - Structured output generation using Gemini API and Pydantic schema enforcement.

### 5.5 FastAPI Deployment & Schema Enforcement
- **Endpoint**: `POST /ask`
  - **Request**: `{"query": "What is the return policy on milk and fresh items?"}`
  - **Response (Pydantic Model)**:
    ```json
    {
      "answer": "Based on Zepto's official policy documentation (policy_1.txt)...",
      "sources": ["policy_1.txt", "policy_7.txt"],
      "confidence": 0.92
    }
    ```
- **Canned Refusal for Off-Topic Queries**:
  - `{"query": "Write a Python script for quicksort"}`
  - `{"answer": "I am Zepto's policy assistant. I can only assist with Zepto delivery, order, return, and service policy questions.", "sources": [], "confidence": 1.0}`

---

## 6. Execution & Setup Instructions

### 6.1 Running Module 1: Data Pipeline
```bash
cd data_pipeline
python run_pipeline.py
```

### 6.2 Running Module 2: Analytics Pipeline
```bash
cd analytics/src
python run_analytics.py
```

### 6.3 Running Module 3: Support Assistant
```bash
cd support_assistant

# Run test suite
python test_assistant.py

# Start FastAPI server
python app/main.py
# Or using uvicorn:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6.4 Running with Docker
```bash
cd support_assistant
docker build -t zepto-support-assistant:latest .
docker run -p 8000:8000 zepto-support-assistant:latest
```

---

## 7. Capstone Verification Checklist

- [x] **Module 1**: Scraped $\ge 60$ books across $\ge 3$ categories with `requests` + `BeautifulSoup`.
- [x] **Module 1**: Star ratings converted to integers (1-5), availability to Boolean, and GBP to INR at 105.50.
- [x] **Module 1**: Normalized SQLite database created (`categories`, `books`) with PK/FK relationships.
- [x] **Module 1**: 5 SQL queries covering `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`, `BETWEEN`, `JOIN`.
- [x] **Module 1**: SQL JOIN (`pd.read_sql`) vs Pandas Merge (`pd.merge`) verified 100% identical.
- [x] **Module 2**: Ingested Titanic data once with `sns.load_dataset('titanic')` and saved to `titanic.csv`.
- [x] **Module 2**: Systematic missing values policy ($<5\%$ drop, $5-30\%$ impute, $>30\%$ drop/flag).
- [x] **Module 2**: Univariate IQR outlier detection, correlation heatmap, and 4-chart survival data story.
- [x] **Module 2**: Stratified train/test split and leakage-free Scikit-Learn `Pipeline` with `ColumnTransformer`.
- [x] **Module 2**: Logistic Regression, Decision Tree (visualized), and Random Forest classifiers.
- [x] **Module 2**: Class imbalance comparison (Baseline vs Class Weights vs SMOTE).
- [x] **Module 2**: Random Forest hyperparameter tuning with `GridSearchCV` and OOB scoring.
- [x] **Module 2**: Multivariate Linear Regression for `fare` and Breusch-Pagan heteroscedasticity test.
- [x] **Module 2**: Model summary table exported to CSV and best pipeline serialized with `joblib`.
- [x] **Module 3**: 8 Zepto policy files created and indexed into ChromaDB with `all-MiniLM-L6-v2`.
- [x] **Module 3**: LangGraph StateGraph workflow with `general_question` (canned refusal) and `policy_question` routes.
- [x] **Module 3**: `MOCK_LLM=1` keyword heuristic & templated response + `MOCK_LLM=0` Gemini path.
- [x] **Module 3**: FastAPI app with `POST /ask` and strict Pydantic JSON schema (`answer`, `sources`, `confidence`).
- [x] **Module 3**: Production Dockerfile and automated test suite.
