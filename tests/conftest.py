import os
import tempfile

# Tests never touch the real database: they use a throwaway SQLite file.
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"