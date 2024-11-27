import redis


def parse_arguments():
    from argparse import ArgumentParser

    prog = ArgumentParser("GW app debug tools.")
    prog.add_argument("--host", dest="redis_host", type=str, default="127.0.0.1")
    prog.add_argument("--port", dest="redis_port", type=int, default=6479)
    prog.add_argument("--db", dest="redis_db", type=int, default=0)

    return prog.parse_args()


if __name__ == "__main__":

    conf = parse_arguments()
    print(f"redis connection: {conf.redis_host}:{conf.redis_port}, db {conf.redis_db}")
    rdb = redis.Redis(host=conf.redis_host, port=conf.redis_port, db=conf.redis_db)
