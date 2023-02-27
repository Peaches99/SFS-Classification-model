import uvicorn
from fastapi import FastAPI, File
import service
import redis

r = redis.Redis(host='redis', port="6379", password="jy972yhkry781rq687yir26")
app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    r.ping()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
