import uvicorn
from fastapi import FastAPI, File, UploadFile
import service


app = FastAPI()
# replace host with redis when using docker


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/upload/")
async def create_upload_file(file: UploadFile):
    """Upload an image to the database."""
    return await service.upload_image(file, "token")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
