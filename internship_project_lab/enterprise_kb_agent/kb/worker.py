import logging
import threading
from .agent import Harness
from .ingestion import ingest


class Workers:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings
        self.stop_event = threading.Event()
        self.threads = []

    def start(self):
        self.db.execute("UPDATE documents SET status='queued' WHERE status='processing'")
        self.db.execute("UPDATE runs SET status='queued' WHERE status='running'")
        for kind in ("documents", "runs"):
            thread = threading.Thread(target=self.loop, args=(kind,), daemon=True, name=f"kb-{kind}")
            thread.start()
            self.threads.append(thread)

    def loop(self, kind):
        while not self.stop_event.is_set():
            try:
                row = self.db.one(f"SELECT id FROM {kind} WHERE status='queued' ORDER BY created_at LIMIT 1")
                if row:
                    if kind == "documents":
                        ingest(self.db, self.settings, row["id"])
                    else:
                        Harness(self.db, self.settings).execute(row["id"])
                else:
                    self.stop_event.wait(.35)
            except Exception:
                logging.exception("Knowledge worker failed")
                self.stop_event.wait(1)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=3)
