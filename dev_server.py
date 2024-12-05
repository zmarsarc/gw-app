from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response
import json

app = FastAPI()

@app.post("/")
async def callback(req: Request):
    resp = await req.body()
    print(json.loads(resp))
    return Response()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, port=9000)