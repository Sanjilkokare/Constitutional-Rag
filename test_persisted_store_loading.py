import json
import unittest
import uuid
from pathlib import Path

import config
import legal_index as li
import storage


class PersistedStoreLoadingTests(unittest.TestCase):
    def test_storage_loader_accepts_bom_documents_json(self):
        root = Path(__file__).resolve().parent
        suffix = uuid.uuid4().hex
        documents = root / f"_tmp_documents_{suffix}.json"

        orig_docs = config.DOCUMENTS_JSON
        orig_index = storage.INDEX_FILE
        orig_meta = storage.META_FILE
        orig_state = (storage._index, storage._metadata, storage._docs)

        try:
            documents.write_text("\ufeff[]\n", encoding="utf-8")
            config.DOCUMENTS_JSON = documents
            storage.INDEX_FILE = root / f"_tmp_index_{suffix}.faiss"
            storage.META_FILE = root / f"_tmp_metadata_{suffix}.json"
            storage._index = None
            storage._metadata = []
            storage._docs = []

            docs = storage.get_doc_records()
            self.assertEqual(docs, [])
            self.assertEqual(storage.get_total_chunks(), 0)
        finally:
            config.DOCUMENTS_JSON = orig_docs
            storage.INDEX_FILE = orig_index
            storage.META_FILE = orig_meta
            storage._index, storage._metadata, storage._docs = orig_state
            documents.unlink(missing_ok=True)

    def test_legal_index_loader_accepts_bom_json(self):
        root = Path(__file__).resolve().parent
        suffix = uuid.uuid4().hex
        legal_index_file = root / f"_tmp_legal_index_{suffix}.json"
        payload = {
            "articles": {"22": [1, 2]},
            "parts": {},
            "schedules": {},
            "lists": {},
            "entries": {},
            "tags": {},
        }

        orig_path = li.LEGAL_INDEX_FILE
        try:
            legal_index_file.write_text("\ufeff" + json.dumps(payload), encoding="utf-8")
            li.LEGAL_INDEX_FILE = legal_index_file
            idx = li.LegalIndex.load()
            self.assertEqual(idx.articles.get("22"), [1, 2])
        finally:
            li.LEGAL_INDEX_FILE = orig_path
            legal_index_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
