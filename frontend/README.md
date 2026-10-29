# Knowledge Studio frontend

Vite + React + TypeScript frontend for the Self-Correcting RAG API.

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The API base URL defaults to `http://localhost:8000` and can be changed with `VITE_API_URL` in `frontend/.env`.

Run the backend and frontend together in two terminals from the project root:

**Terminal 1 — backend**

```powershell
.\venv\Scripts\python.exe -m uvicorn app.api:app --port 8000
```

**Terminal 2 — frontend**

```powershell
cd frontend
npm install
npm run dev
```

To try the interface without LLM quota, set `MOCK_LLM=1` in the backend terminal before starting Uvicorn:

```powershell
$env:MOCK_LLM = '1'
.\venv\Scripts\python.exe -m uvicorn app.api:app --port 8000
```

## Checks

```powershell
npm test
npm run build
```

The API enables CORS only for the Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`.
