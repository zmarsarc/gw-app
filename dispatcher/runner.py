from argparse import ArgumentParser

from flask import Flask
from model_runner import ModelRunner, ModelError
from loguru import logger

parser = ArgumentParser()
parser.add_argument("model_name")
parser.add_argument("-p", dest="port", type=int)


app = Flask(__name__)


@app.get("/state")
def get_model_state():
    pass


@app.post("/task")
def start_inference_task():
    pass


def main():
    arguments = parser.parse_args()
    
    try:
        runner = ModelRunner(arguments.model_name)
    except ModelError as e:
        logger.error(f"load model error, {e}")
        return
    

    app.run(host='127.0.0.1',
            port=arguments.port if arguments.port is not None else 8000)


if __name__ == '__main__':
    main()
