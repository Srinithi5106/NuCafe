"""
locustfile.py — NuCafé Load Test
Test SQLite app and Neon app side by side.

Step 1: Run SQLite app
    streamlit run app_sqlite.py --server.port 8501

Step 2: Run Neon app  
    streamlit run app.py --server.port 8502

Step 3: Run locust against SQLite
    locust -f locustfile.py --host=http://localhost:8501

Step 4: Screenshot results, then run against Neon
    locust -f locustfile.py --host=http://localhost:8502

Compare the failure rates and response times.
"""

from locust import HttpUser, task, between
import random

class NuCafeUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def load_home(self):
        self.client.get("/")

    @task(2)
    def browse_category(self):
        cat = random.choice(["South+Indian", "Biryani", "Fastfood", "Beverage", "Snack"])
        self.client.get(f"/?category={cat}", name="Browse Category")

    @task(1)
    def search(self):
        q = random.choice(["dosa", "biryani", "coffee", "burger", "samosa"])
        self.client.get(f"/?search={q}", name="Search")