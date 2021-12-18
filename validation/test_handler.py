from redis_storage import RedisStorage

s = RedisStorage()
payload = {"path": "images/other/1002.jpg"}
s.save(payload)
s.publish(RedisStorage.INPUT_VALIDATION, payload)

payload = {"path": "images/glasses/10.jpg"}
s.save(payload)
s.publish(RedisStorage.INPUT_VALIDATION, payload)