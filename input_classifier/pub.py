import os
import redis
import sys
import json
 
redis_conn = redis.Redis(charset="utf-8", decode_responses=True)
 
def pub():
    data = {"message": "hello", "from": "xxx", "to": "YOUR_NUMBER"}
    redis_conn.rpush('queue:my', json.dumps(data))
    return {}

if __name__ == "__main__":
    pub()