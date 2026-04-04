# NuCafé

> AI-powered hybrid food ordering & recommendation system for university campuses

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![PostgreSQL](https://img.shields.io/badge/Neon-PostgreSQL-green) ![Docker](https://img.shields.io/badge/Docker-20.10+-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

NuCafé replaces static, scrollable menus with a real-time AI layer. It personalises every student's view of a 120+ item menu, surfaces trending dishes, predicts kitchen wait times, and suggests complementary items — all within a **186 ms** median response window.

| Metric | Value |
|---|---|
| Throughput | 109 RPS |
| Median latency | 186 ms |
| P95 latency | 242 ms |
| Failure rate @ 10,000 users | 0% |
| Menu items ranked | 120+ |

---

## Features

- **Hybrid recommendations** — 50% collaborative filtering + 30% content-based DNA + 20% time-decay trending, blended into a single "Match %" score
- **Trending now** — exponential decay `W(t) = exp(−0.05·Δt)` surfaces what's actively being ordered in the kitchen right now
- **Smart cart pairing** — Apriori association mining (support ≥ 2%, confidence ≥ 50%) suggests complementary sides and beverages at checkout
- **Wait-time filter** — Random Forest Regressor (100 trees) predicts prep time; students filter by "ready within X minutes" to align orders with their schedule
- **Flavour DNA** — a per-user JSONB affinity profile that evolves silently after every purchase, solving the cold-start problem for new menu items

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/nucafe.git
cd nucafe
pip install -r requirements.txt

# 2. Set environment variables
export DATABASE_URL="postgresql://user:password@neon-host/nucafe"
export SECRET_KEY="your-secret-key"

# 3. Train models (generates .pkl files)
python train_ai.py

# 4. Run locally
streamlit run app.py

# 5. Or run via Docker
docker build -t nucafe .
docker run -p 8501:8501 --env-file .env nucafe
```

---

## System Architecture

```
Streamlit UI  →  ML Middleware  →  Neon PostgreSQL  →  Docker / Railway
```

The ML middleware has three independent sub-services:

- **Recommendation Service** — Hybrid 50-30-20 blend (Matrix Factorization, Category Affinity, Time-Decay)
- **Association Rule Service** — Apriori algorithm on the current cart for real-time pairing suggestions
- **Wait-Time Prediction Service** — Random Forest Regressor evaluating live kitchen load

Pre-trained `.pkl` weights are bundled into the Docker image so dot-product inference runs at request time without retraining.

---

## ML Models

### 1. Matrix Factorization (Collaborative Filtering)
Decomposes the user-item interaction matrix into two low-rank matrices using Stochastic Gradient Descent.

- Latent features: `k = 12`
- Learning rate: `α = 0.0002`
- Regularisation: `β = 0.02`
- Epochs: `20`
- User matrix `P`: 1000 × 12 | Item matrix `Q`: 126 × 12

### 2. Profile DNA (Content-Based Filtering)
A 6-element category affinity vector `[South Indian, North Indian, Biryani, Beverage, Fastfood, Snack]` that updates after every order using an incremental learning rate of `0.15`. Solves the cold-start problem for newly added menu items.

### 3. Time-Decay Trending
```
W(t) = exp(−0.05 · Δt)
```
Aggregates decay weights across a 60-minute rolling window and normalises scores to `[0, 1]`. An order placed 14 minutes ago carries roughly half the weight of one placed just now.

### 4. Hybrid Blend
```
Final Score = 0.5 × CF + 0.3 × Content + 0.2 × Trending
```

### 5. Apriori Association Mining
- Minimum support: `0.02`
- Minimum confidence: `0.50`
- Trained on full transactional history; rules are pre-computed and queried at checkout

### 6. Random Forest Wait-Time Regressor
- Estimators: `100 decision trees`
- Features: `base_prep_time`, `queue_depth`, `hour_of_day`, `weekend_flag`
- Trained on 5,000 synthetic records simulating rush-hour dynamics (12:00–14:00)

---

## Database Schema

```sql
User          -- id (UUID PK), username, flavour_dna (JSONB), latent_vector (FLOAT[12])
Menu_Item     -- id (SERIAL PK), name, category (B-Tree indexed), base_prep (INT), latent_vector (FLOAT[12])
Order         -- id, user_id (FK), order_time (TIMESTAMP, B-Tree indexed), status
Order_Item    -- id, order_id (FK), item_id (FK)
Kitchen_Queue -- id, order_id (FK), est_wait (INT — RF prediction output)
```

**Design decisions:**
- `JSONB` for Flavour DNA allows schema-free evolution as new food categories are added
- `FLOAT[12]` native arrays enable match-score calculation via a single dot-product query (`< 200 ms`)
- B-Tree indexes on `order_time` and `category` power the 15-minute trending window without full-table scans
- PostgreSQL MVCC provides row-level versioning so reads never block writes during peak load

---

## SQLite vs Neon PostgreSQL

| Metric | SQLite | Neon PostgreSQL |
|---|---|---|
| Max stable users | 5,000 (crash) | 10,000+ (stable) |
| Throughput | 12 RPS | 109 RPS |
| Median latency | High (lock-wait) | 186 ms |
| Failure rate @ 5,000 users | 20.35 / sec | 0% |
| Locking mechanism | Full-database write lock | Row-level MVCC |
| Storage | Local disk (ephemeral) | Multi-AZ distributed |
| Scalability | Vertical only | Horizontal + serverless |
| Recommended for | Dev / prototyping | Production |

SQLite's `BEGIN IMMEDIATE` lock blocks all reads the moment a write is in progress. Under Neon's MVCC, writers create a new row version while readers continue on the previous snapshot — eliminating contention entirely.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.28+ |
| ML | scikit-learn, NumPy, Pandas, MLXtend |
| ORM | SQLAlchemy + Psycopg2 |
| Database | Neon Serverless PostgreSQL |
| Auth | Bcrypt / SHA-256 (salted) |
| Containerisation | Docker 20.10+ |
| CI/CD | Railway (GitHub-triggered deploys) |
| Load testing | Locust |

---

## Scalability Results

Load tested with Locust simulating 10,557 concurrent users at 50 users/sec spawn rate.

| Concurrent users | Avg latency | Failure rate |
|---|---|---|
| 500 | 110 ms | 0% |
| 2,500 | 145 ms | 0% |
| 5,000 | 170 ms | 0% |
| 10,000+ | 186 ms | 0% |

Infrastructure details from the final run:
```
[2026-03-15 13:05] Result: 0 failures, 109.4 RPS achieved.
[2026-03-15 13:05] Response Time: Median 186ms | P95 242ms | P99 310ms
[2026-03-15 13:05] Database (Neon): CPU 0.82 Load, 42 active connections.
```

---

## Hardware Requirements

**Development machine**
- CPU: Quad-core Intel i5 / AMD Ryzen 5 or higher
- RAM: 8 GB minimum (16 GB recommended)
- Storage: 256 GB SSD

**Cloud deployment (Railway / AWS)**
- vCPU: 1+
- RAM: 1–2 GB
- Bandwidth: 100 Mbps+

---

## Known Limitations

- **Cold start** — Collaborative Filtering requires a minimum order history for brand-new users; the DNA model partially compensates via category affinity
- **Cross-region latency** — Neon (cloud) + Railway may introduce occasional P95 spikes from network hops between regions
- **Wait-time accuracy** — Random Forest predictions depend on kitchen staff updating order statuses promptly; stale statuses degrade accuracy

---

## Roadmap

- [ ] Computer Vision queue detection — overhead cameras for physical queue length instead of status-update polling
- [ ] Group recommendations — suggest a single platter or combo based on the collective Flavour DNA of a friend group
- [ ] NLP voice ordering — hands-free interface for students multitasking between lectures

---
## Login 
<img width="416" height="407" alt="01_login" src="https://github.com/user-attachments/assets/7fa17bc0-7246-4678-92ca-a2322c5cacf6" />

## Top Picks for You
<img width="809" height="365" alt="n1" src="https://github.com/user-attachments/assets/9c4480f5-8c58-454a-8b96-8935f89b3c79" />

## Trending Now
<img width="806" height="344" alt="n2" src="https://github.com/user-attachments/assets/1b8b486f-c3a4-4c2a-8710-9ddd859a568a" />

## Pairs Well with
<img width="806" height="386" alt="n3" src="https://github.com/user-attachments/assets/d18b7486-32a6-434a-9c19-0c3c9afadbbe" />

## Order by Available Time
<img width="801" height="389" alt="n4" src="https://github.com/user-attachments/assets/44b09b4b-db85-4641-bb5f-89956012ef93" />

## References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer, 42(8), 30–37.
2. Agrawal, R., & Srikant, R. (1994). *Fast Algorithms for Mining Association Rules*. VLDB, 487–499.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
4. Burke, R. (2002). *Hybrid Recommender Systems: Survey and Experiments*. UMUAI, 12(4), 331–370.
5. PostgreSQL Global Development Group (2024). *Chapter 13: Concurrency Control — MVCC*. PostgreSQL 16 Documentation.
6. Merkel, D. (2014). *Docker: Lightweight Linux Containers for Consistent Development and Deployment*. Linux Journal.
7. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.

---

## License

MIT License — built for university campus environments.
