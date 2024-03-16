import lmdb
import pickle
from typing import Any


class LMDBDataHandler:
    def __init__(self, path: str, read_only: bool = False, write_only: bool = False):
        self.env = lmdb.open(
            path,
            max_dbs=1,
            readonly=read_only,
            create=not read_only,
            map_size=int(1e12),
            lock=False,
        )
        self.db = self.env.open_db(b"main")
        self.read_only = read_only
        self.write_only = write_only

        if read_only and write_only:
            raise ValueError("Cannot set both read_only and write_only to True.")

    def __getitem__(self, index: str) -> Any:
        if self.write_only:
            raise PermissionError("Database is in write-only mode.")

        with self.env.begin(db=self.db, write=False) as txn:
            value = txn.get(index.encode("utf-8"))
            if value is None:
                raise KeyError(f"index {index} not found.")
            return pickle.loads(value)

    def append(self, index: str, value: Any):
        if self.read_only:
            raise PermissionError("Database is in read-only mode.")

        with self.env.begin(db=self.db, write=True) as txn:
            txn.put(index.encode("utf-8"), pickle.dumps(value))

    def shape(self) -> int:
        with self.env.begin(db=self.db, write=False) as txn:
            return txn.stat()["entries"]
