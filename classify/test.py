import postgresql
import redis

r = redis.StrictRedis(host='10.100.10.6', port=6379, db=0)
