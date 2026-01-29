from fastapi import FastAPI

app = FastAPI(title="MarketSentinel")

@app.get("/")
def root():
    return {"message": "MarketSentinel backend is running"}
