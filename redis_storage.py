import json
import os
import redis
import signal
import sys
import time

from uuid import uuid4

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

class RedisStorage:
    INPUT_VALIDATION = "INPUT_VALIDATION"
    SEGMENTATION = "SEGMENTATION"

    def __init__(self):
        host = os.environ.get("REDIS_HOST", 'localhost')

        self.redis_conn = redis.Redis(
            charset="utf-8",
            decode_responses=True,
            host=host,
        )

    def publish(self, queue_name, payload):
        self.redis_conn.rpush(f"queue:{queue_name}", json.dumps(payload))
        return {}

    def subscribe(self, queue_name, handler):
        print(f"Subscribed to `{queue_name}`")

        killer = GracefulKiller()
        while not killer.kill_now:
            packed = self.redis_conn.blpop([f"queue:{queue_name}"], 1)

            if not packed:
                continue

            payload = json.loads(packed[1])
            print(f"Queue `{queue_name}` recieved: `{payload}`")
            handler(payload)

        print(f"Unsubscribed from `{queue_name}`")

    def save(self, payload):
        uid = str(uuid4())
        payload["uid"] = uid
        return self.update(uid, payload)

    def load(self, uid):
        payload = self.redis_conn.get(uid)

        if payload:
            return json.loads(payload)

    def update(self, uid, payload):
        self.redis_conn.set(uid, json.dumps(payload))
        return payload
