from redis import Redis
from rq import SimpleWorker, Queue

conn = Redis(host='localhost', port=6379)
q = Queue('training', connection=conn)
worker = SimpleWorker([q], connection=conn)
worker.work()