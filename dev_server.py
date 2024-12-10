
from typing import Optional

from argparse import ArgumentParser


def parse_cli_arguments():

    parser = ArgumentParser()
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0")
    parser.add_argument("--port", dest="port", type=int, default=9000)

    return parser.parse_args()


class Parser:

    def __init__(self):
        self._buf = []

    def lookahead(self, n: int = 1) -> Optional[str]:
        while len(self._buf) == 0:
            self._read_more()

        if 0 < n <= len(self._buf):
            return self._buf[n-1]
        else:
            return None

    def consume(self, n: int = 1):
        while n >= 0:
            if len(self._buf) == 0:
                return
            self._buf.pop(0)
            n -= 1

    def _read_more(self):
        ipt = input("> ")
        self._buf.extend(ipt.strip().split())


if __name__ == '__main__':
    pass