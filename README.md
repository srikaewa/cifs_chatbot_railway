# Forensic ChatBot Template (FastAPI + Railway-ready)

## Start locally
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Deploy to Railway
Set Start Command:
```
uvicorn main:app --host 0.0.0.0 --port=$PORT
```