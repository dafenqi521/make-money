### Task 1: Project Scaffold & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/ui/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `requirements.txt` with pinned versions

- [ ] **Step 1: Create directory structure and `__init__.py` files**

```bash
New-Item -ItemType Directory -Force -Path "F:\MyProject\src\data"
New-Item -ItemType Directory -Force -Path "F:\MyProject\src\ui"
New-Item -ItemType Directory -Force -Path "F:\MyProject\tests"
New-Item -ItemType File -Force -Path "F:\MyProject\src\__init__.py"
New-Item -ItemType File -Force -Path "F:\MyProject\src\data\__init__.py"
New-Item -ItemType File -Force -Path "F:\MyProject\src\ui\__init__.py"
New-Item -ItemType File -Force -Path "F:\MyProject\tests\__init__.py"
```

- [ ] **Step 2: Write `requirements.txt`**

```txt
streamlit>=1.28.0
akshare>=1.12.0
pandas>=2.0.0
plotly>=5.17.0
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without errors.

- [ ] **Step 4: Verify AKShare works**

```bash
python -c "import akshare as ak; df = ak.fund_etf_hist_sina(symbol='510300'); print(df.tail(3))"
```

Expected: Prints last 3 rows of 沪深300 ETF historical data.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: project scaffold and dependencies for v0.1"
```

---

