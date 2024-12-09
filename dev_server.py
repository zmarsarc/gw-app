import json
import sys
import threading


def parse_cli_arguments():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0")
    parser.add_argument("--port", dest="port", type=int, default=9000)

    return parser.parse_args()


class DebugCallbackServer:

    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        try:
            from fastapi import FastAPI
            from fastapi.requests import Request
            from fastapi.responses import Response
        except ImportError:
            print(
                "Imput fastapi not found, please install by command 'pip install fastapi'")
            sys.exit(1)

        self.host = host
        self.port = port

        self.app = FastAPI()

        async def callback(req: Request):
            resp = await req.body()
            print(json.loads(resp))
            return Response()
        self.app.post("/")(callback)

    def run(self):
        try:
            import uvicorn
        except ImportError:
            print("Import uvicor not found, please install by 'pip install uvicor'")

        t = threading.Thread(target=uvicorn.run, kwargs={
            "app": self.app,
            "host": self.host,
            "port": self.port},
            daemon=True)
        t.start()


def command_loop():

    try:
        import requests
    except ImportError:
        print("Import package requests not found, try 'pip install requests'")
        sys.exit(1)

    while True:
        ipt = input("press enter to send a request to start a task.")
        
        requests.post("http://localhost:8000/task", json={
            "mid": "yolov8n-det",
            "image_url": "/app/data/bus.jpg",
            "post_process": "none",
            "callback": "host.docker.internal:9000/"
        })


if __name__ == '__main__':

    args = parse_cli_arguments()

    server = DebugCallbackServer(host=args.host, port=args.port)
    server.run()

    command_loop()
