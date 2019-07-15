import time

class PipelineTimer:
    def __init__(self):
        self.total_time = 0
        self.timer_start = 0
        self.timer_end = 0
        self.record = []
    def start(self):
        self.timer_start = time.time()
    def end(self):
        self.timer_end = time.time() - self.timer_start
        self.record.append(self.timer_end)
    def report(self):
        return sum(self.record)/len(self.record)
