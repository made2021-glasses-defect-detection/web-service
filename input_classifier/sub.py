import os
import redis
import json
import signal
import time

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

def subscribe(name: str):
    redis_conn = redis.Redis(charset="utf-8", decode_responses=True)
    killer = GracefulKiller()
    while not killer.kill_now:
        packed = redis_conn.blpop(['queue:my'], 1)

        if not packed:
            continue

        data = json.loads(packed[1])

        print(data)

    print("Interrupted")

if __name__ == "__main__":
    # Process(target=sub, args=("reader1",)).start()
    sub("test")