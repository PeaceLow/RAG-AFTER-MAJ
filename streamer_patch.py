import re
from transformers.generation.streamers import TextIteratorStreamer

class CleanStreamer(TextIteratorStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_think = False
        self._buffer = ""
        import queue
        self._clean_queue = queue.Queue()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._buffer += text
        
        if not self._in_think and "<think>" in self._buffer:
            parts = self._buffer.split("<think>", 1)
            if parts[0]:
                self._clean_queue.put(parts[0])
            self._in_think = True
            self._buffer = "<think>" + parts[1]

        if self._in_think and "</think>" in self._buffer:
            parts = self._buffer.split("</think>", 1)
            self._in_think = False
            self._buffer = parts[1].lstrip('\n')

        if not self._in_think:
            if self._buffer:
                self._clean_queue.put(self._buffer)
                self._buffer = ""

        if stream_end:
            self._clean_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self._clean_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        return value
