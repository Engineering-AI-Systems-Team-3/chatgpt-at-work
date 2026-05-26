import time
import threading


class RateLimiter:
    """
    Tocken bucket rate limiter to control the rate of API calls across multiple threads.
    """

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_check = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_check
            self.last_check = now
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)

            if self.tokens >= 1:
                self.tokens -= 1
            else:
                wait = (1 - self.tokens) / self.rate
                time.sleep(wait)
                self.tokens = 0
