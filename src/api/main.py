from .app import app

# Esta línea permite correr la app directamente con 'python -m src.api.main'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)