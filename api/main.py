import uvicorn
from fastapi import FastAPI, File
import service
import redis

r = redis.Redis(host="0.0.0.0", port="6379", password="jy972yhkry781rq687yir26")

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/set/{a}/{b}")
async def set(a: int, b: int):
    r.set(a, b)

@app.get("/list")
async def listAll():
    return r.keys()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
