"""Service layer for the API."""

import redis
import uuid

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

r = redis.Redis(host="localhost", port="6379", password="jy972yhkry781rq687yir26")
r.ping()


async def authorize(token):  # TODO implement
    return True


async def generate_id():
    """Generate a unique ID."""
    return str(uuid.uuid4())


async def is_image(filename):
    """Check if the file is a valid image."""
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS


async def upload_image(file, token):
    """Upload an image to the database."""
    if not await authorize(token):
        return {"error": "Token is invalid."}

    print(file.filename)
    if not await is_image(file.filename):

        return {"error": "File is not an image."}

    image_id = await generate_id()

    r.set(file.filename, file.file.read())
    return {"success": image_id}
