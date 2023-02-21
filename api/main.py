import uvicorn
from fastapi import FastAPI, File
import service
import redis

r = redis.Redis(host="0.0.0.0", port="6379", password="testing")

r.set("a", 5)

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/set/{a}/{b}")
async def set(a: int, b: int):
    r.set("a", a)
    return {"a": a, "b": b}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
