import http.server
import json
import socketserver
import sys
import threading
from typing import Optional

# prompt_toolkit package is use to build all command line app.
# Install via pip by command 'pip install prompt_toolkit'
try:
    import prompt_toolkit as pt
    from prompt_toolkit.completion import NestedCompleter
except ImportError:
    print("no package 'prompt_toolkit', try 'pip install prompt_toolkit'")
    sys.exit(1)

# redis is required by many gw package, if not gw will not work.
# Install via pip by command 'pip install redis'.
# Or use hiredis by 'pip install redis[hiredis]'.
try:
    import redis
except ImportError:
    print("no package 'redis', try 'pip install redis'")
    sys.exit(1)

# requests package is a popular http package to send request.
# It will use to send task requests in our app.
# Install via pip by command 'pip install requests'.
try:
    import requests
except ImportError:
    print("no package 'requests', try 'pip install requests'")
    sys.exit(1)


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

            pt.print_formatted_text(f"[POST {self.path}] {json.loads(content)}")

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
                (self.host, self.port), self.Handler, bind_and_activate=False
            )
            self._httpd.allow_reuse_address = True
            self._httpd.server_bind()
            self._httpd.server_activate()
            threading.Thread(target=self._httpd.serve_forever).start()

    # Close current running server if have, and reopen new server.
    # If host or port are given, update server host and port.
    def reopen(self, host: str = None, port: int = None):
        self.shutdown()
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port

        self.serve()

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

    def reconnect(self, host: str = None, port: int = None, db: int = None):
        if host:
            self.host = host
        if port:
            self.port = port
        if db:
            self.db = db

        # close current connection.
        self.close()
        self._redis = redis.Redis(host=self.host, port=self.port, db=self.db)

    def close(self):
        if self._redis is not None:
            self._redis.close()
            self._redis = None


# Input parse use to handle cli input.
# It basicly a LL(n) lexer which provide methods to look ahead n tokens.
#
# read_line() to read new line in and parse to tokens.
# lookahead to check the first n token.
# consume will remove first n token from buffer.
class InputParser:

    def __init__(self, status_bar=None, completer=None):
        self._buf = []
        self._status_bar = status_bar
        self._completer = completer
        self._session = pt.PromptSession()

    def lookahead(self, n: int = 1) -> Optional[str]:
        if 0 < n <= len(self._buf):
            return self._buf[n - 1]
        else:
            return None

    def consume(self, n: int = 1):
        while n > 0 and len(self._buf) != 0:
            self._buf.pop(0)
            n -= 1

    def read_input(self, prompt: str = "") -> bool:
        text = self._session.prompt(
            f"(gw devtools){prompt} > ",
            bottom_toolbar=self._status_bar,
            completer=self._completer,
        )
        self._buf.extend(text.strip().split())
        return True

    def clean_buf(self):
        self._buf.clear()

    @property
    def rest_token_num(self) -> int:
        return len(self._buf)


# Make a callable object which use to build status str.
# Used by prompt to show on status bar.
def make_status_bar(rdb: RedisClient, srv: DevServer):
    def maker():
        if rdb.is_connected:
            redis_status = f"<green>connected ({rdb.host}:{rdb.port}@{rdb.db})</green>"

        else:
            redis_status = "<yellow>Redis disconnected</yellow>"

        if srv.running:
            srv_status = f"<green>Dev server running ({srv.host}:{srv.port})</green>"
        else:
            srv_status = "<yellow>Dev server stopped</yellow>"

        return pt.HTML(f"{redis_status} | {srv_status}")

    return maker


# Devsrv commands use to control dev server.
# Can change host and port by command reopen.
def devsrv_commands(parser: InputParser, srv: DevServer):
    if parser.lookahead() == "stop":
        srv.shutdown()
    elif parser.lookahead() == "reopen":
        parser.consume()

        # check if given host and port.
        host = None
        port = None
        if parser.lookahead() is not None:

            # parse address by ":".
            address = parser.lookahead().strip().split(":")
            host = address[0]
            if host == "":
                host = None
            if len(address) > 1:
                port = int(address[1])
            else:
                port = None

        srv.reopen(host, port)

    else:
        pt.print_formatted_text(
            "devsrv usage:\n"
            + "  devsrv stop: stop current running dev server.\n"
            + "  devsrv reopen: reopne dev server by latest host and port.\n"
            + "  devsrv reopne [host[:port]]: reopen dev server by given host and port."
        )

    # Consume token in buffer when accept any commands.
    parser.clean_buf()


# Req commands use to send task request to gw host.
# If host address not given, default localhost:8000
def req_commands(parser: InputParser, srv: DevServer):
    if parser.rest_token_num >= 2:
        model_id = parser.lookahead(1)
        image_path = parser.lookahead(2)
        callback = parser.lookahead(3)
        address = parser.lookahead(4)

        # if address not given, use default localhost:8000
        if not address:
            address = "localhost:8000"

        if not callback:
            callback = f"http://host.docker.internal:{srv.port}"

        if address and model_id and image_path:
            try:
                # build request body.
                req_body = {
                    "mid": model_id,
                    "image_url": image_path,
                    "post_process": "none",
                    "callback": callback,
                }

                # sending request to address.
                pt.print_formatted_text(f"send request to {address}, body: {req_body}")
                resp = requests.post(f"http://{address}/task", json=req_body, timeout=1)

                # handle response, if not ok, print status code.
                # else print response content, let's just assume it will be a json.
                if resp.status_code != 200:
                    pt.print_formatted_text(f"status code {resp.status_code}")
                else:
                    pt.print_formatted_text(
                        f"receive response: {json.loads(resp.content)}"
                    )

            except requests.ConnectionError:
                pt.print_formatted_text(
                    pt.HTML(
                        f'<b fg="red">connect error, host {address} not reachable.</b>'
                    )
                )
            except requests.ReadTimeout:
                pt.print_formatted_text(
                    f"wait response timeout, host {address} may occure error or not reachable."
                )

            return parser.clean_buf()

    # Print req command usage message if command invalid.
    pt.print_formatted_text(
        "req usage:\n"
        + "  req model_id image_path [callback] [host address]: send task request to task server.\n"
        + "it will use dev server as callback."
    )
    return parser.clean_buf()


# Redis commands use to manage use connection.
# Also it can send redis command via connection.
def redis_commands(parser: InputParser, rdb: RedisClient):

    # check if redis connected first, if not, prompt to connect.
    if not rdb.is_connected and parser.lookahead() != "connect":
        pt.print_formatted_text(
            pt.HTML(
                '<b fg="yellow">redis not connected, try command redis reconnect first.</b>'
            )
        )
        return parser.clean_buf()

    # disconnect current redis connection.
    if parser.lookahead() == "disconnect":
        rdb.close()
        return parser.clean_buf()

    # connect redis, can give new host port.
    if parser.lookahead() == "connect":
        parser.consume()

        # parse address if have.
        host = None
        port = None
        if parser.lookahead() is not None:
            address = parser.lookahead().strip().split(":")

            # if just given port, host will be empti str.
            host = address[0]
            if host == "":
                host = None

            # if have port, convert to int.
            if len(address) > 1:
                port = int(address[1])
            else:
                port = None

        # reconnect redis, will close current connection.
        rdb.reconnect(host=host, port=port)
        return parser.clean_buf()

    # list all keys in current redis db.
    # Support given pattern by extral argument.
    if parser.lookahead() == "keys":
        pattern = parser.lookahead(2)
        if not pattern:
            pattern = "*"
        rdb.client.keys(pattern)

        return parser.clean_buf()

    # flush currenet db.
    if parser.lookahead() == "flushdb":
        # notic that it is a dengours operation.
        parser.clean_buf()
        pt.print_formatted_text(
            pt.HTML(
                '<b fg="red">this operation will remove all keys in this db, please comfirme.</b>'
            )
        )

        # require yes or no.
        while parser.read_input("flush db? [y/n]"):
            if parser.lookahead() in ["y", "yes", "Y"]:
                rdb.client.flushdb()
                return parser.clean_buf()
            if parser.lookahead() in ["n", "no"]:
                return parser.clean_buf()

            # clean invalid input.
            parser.clean_buf()


# Command completer for prompt.
completer = NestedCompleter.from_nested_dict(
    {
        "devsrv": {"stop": None, "reopen": None},
        "req": {"yolov8n-det": None, "yolov8n-cls": None},
        "redis": {"disconnect": None, "keys": None, "connect": None, "flushdb": None},
        "exit": None,
        "quit": None,
        "clear": None,
        "cls": None,
    }
)


def main():

    # Make redis connect.
    rdb = RedisClient()

    # Start dev server.
    srv = DevServer()
    srv.serve()

    # Make new parser to handle cli input.
    parser = InputParser(status_bar=make_status_bar(rdb, srv), completer=completer)

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

        # Devserver commands.
        if parser.lookahead() == "devsrv":
            parser.consume()
            devsrv_commands(parser, srv)
            continue

        # Request commands.
        if parser.lookahead() == "req":
            parser.consume()
            req_commands(parser, srv)
            continue

        # Redis commands.
        if parser.lookahead() == "redis":
            parser.consume()
            redis_commands(parser, rdb)
            continue

        # No support command finded, clean buffer and try next line.
        parser.clean_buf()

    # Main loop stopped.
    # Do cleanup.
    srv.shutdown()
    rdb.close()


if __name__ == "__main__":
    main()
