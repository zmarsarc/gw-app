import http.server
import json
import socketserver
import threading
from typing import Optional

import prompt_toolkit as pt
import redis
import redis.connection


# A simple http server provide a post endpoint.
# Which use to test task callbask.
# The default host is localhost, and default port is 9000
#
# Call method serve() to start http server, it will run in backgfound thread.
# Don't forget call shutdown() after use.
class DevServer:

    class Handler(http.server.SimpleHTTPRequestHandler):

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            content = self.rfile.read(content_length)
            self.rfile.close()

            pt.print_formatted_text(
                f"receive post request: {json.loads(content)}")

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write("ok".encode())

        # Keep this method empty to prevent server print log.
        def log_request(self, code="-", size="-"):
            pass

    def __init__(self, host="127.0.0.1", port=9000):
        self._httpd = None
        self.host = host
        self.port = port

    def serve(self):
        if self._httpd is not None:
            return
        else:
            self._httpd = socketserver.TCPServer(
                (self.host, self.port), self.Handler, bind_and_activate=False)
            self._httpd.allow_reuse_address = True
            self._httpd.server_bind()
            self._httpd.server_activate()
            threading.Thread(target=self._httpd.serve_forever).start()

    def shutdown(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd.socket.close()
            self._httpd = None

    @property
    def running(self) -> bool:
        return self._httpd is not None


# Redis connection manager to manage connection and status.
# Provide reconnect() method to connect new redis by give host, port and db.
#
# Defulat redis is 127.0.0.1:6379 and use db 0.
#
# Don't forget call close() after use.
class RedisClient:

    def __init__(self, host="127.0.0.1", port=6379, db=0):
        self.host = host
        self.port = port
        self.db = db

        self._redis = redis.Redis(host=host, port=port, db=db)

    @property
    def client(self) -> redis.Redis:
        return self._redis

    @property
    def connection_pool(self) -> redis.Connection:
        return self._redis.connection_pool

    @property
    def is_connected(self) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except redis.ConnectionError:
            return False

    def reconnect(self):
        self._redis = redis.Redis(host=self.host, port=self.port, db=self.db)

    def close(self):
        if self._redis is not None:
            self._redis.close()


# Input parse use to handle cli input.
# It basicly a LL(n) lexer which provide methods to look ahead n tokens.
#
# read_line() to read new line in and parse to tokens.
# lookahead to check the first n token.
# consume will remove first n token from buffer.
class InputParser:

    def __init__(self, status_bar=None):
        self._buf = []
        self._status_bar = status_bar

    def lookahead(self, n: int = 1) -> Optional[str]:
        if 0 < n <= len(self._buf):
            return self._buf[n-1]
        else:
            return None

    def consume(self, n: int = 1):
        while n > 0 and len(self._buf) != 0:
            self._buf.pop(0)
            n -= 1

    def read_input(self, prompt: str = "") -> bool:
        text = pt.prompt(
            f"(gw devtools){prompt} > ", bottom_toolbar=self._status_bar)
        self._buf.extend(text.strip().split())
        return True

    def clean_buf(self):
        self._buf.clear()


def main():

    # Make new parser to handle cli input.
    parser = InputParser()

    # Make redis connect.
    rdb = RedisClient()

    # Start dev server.
    srv = DevServer()
    srv.serve()

    # Into main loop.
    while parser.read_input():

        # Handle exit command.
        if parser.lookahead() in ["exit", "quit"]:
            break

        # Clean screen command.
        if parser.lookahead() in ["cls", "clear"]:
            parser.consume()
            pt.shortcuts.clear()
            continue

        # Redis commands.
        if parser.lookahead() == "redis":
            continue

        # Task commands;
        if parser.lookahead() == "task":
            continue

        # No support command finded, clean buffer and try next line.
        parser.clean_buf()

    # Main loop stopped.
    # Do cleanup.
    srv.shutdown()
    rdb.close()


if __name__ == "__main__":
    main()
