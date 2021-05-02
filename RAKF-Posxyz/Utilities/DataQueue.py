import threading


class DataQueue:
    def __init__(self):
        self.queue = []
        self._lock = threading.Lock()

    def pop(self):
        if self._lock.acquire( blocking=True, timeout=0.1 ):
            temp = self.queue.pop()
            self._lock.release()
            return {"data": temp, "status": True, "reason": " "}
        return {"data": None, "status": False, "reason": "Failed to acquire the lock"}

    def append(self, data):
        if self._lock.acquire( blocking=True, timeout=0.1 ):
            self.queue.append( data )
            self._lock.release()
            return {"status": True, "reason": " "}
        return {"status": False, "reason": "Failed to acquire the lock"}

    def __len__(self):
        return self.queue.__len__()

    def is_empty(self):
        return True if self.queue.__len__() <= 0 else False


measurement_queue = DataQueue()
