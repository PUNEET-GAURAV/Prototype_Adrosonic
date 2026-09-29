"""Thin Qdrant wrapper: collection design, payload indexes, upsert/delete, pre-retrieval filters."""
from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

NS = uuid.UUID("6f1c2a52-5d3b-4d8e-9a55-0a1f4c2e7b10")


def point_id(passage_id: str) -> str:
    """Deterministic UUID from the human-readable passage id (idempotent upserts, easy deletes)."""
    return str(uuid.uuid5(NS, passage_id))


def build_filter(categories=None, source=None, origin=None, date_from=None, date_to=None):
    """Build a Qdrant filter. Applied INSIDE the vector database, to every retrieval leg."""
    must = []
    if categories:
        must.append(models.FieldCondition(key="category", match=models.MatchAny(any=list(categories))))
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    if origin:
        must.append(models.FieldCondition(key="origin", match=models.MatchValue(value=origin)))
    if date_from is not None or date_to is not None:
        must.append(models.FieldCondition(key="ingested_at", range=models.Range(gte=date_from, lte=date_to)))
    return models.Filter(must=must) if must else None


class Store:
    def __init__(self, url=None, collection="verity", prefer_grpc=False, grpc_port=6334, location=None, path=None):
        self.collection = collection
        if path:
            self.client = QdrantClient(path=path)
        elif location:
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc, grpc_port=grpc_port, timeout=120)

    def create(self, dim: int, recreate: bool = False) -> None:
        if self.client.collection_exists(self.collection):
            if not recreate:
                return
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            # IDF modifier: Qdrant computes IDF at query time from the live corpus.
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Payload indexes created BEFORE bulk upload so filtered HNSW search stays fast.
        for f in ("category", "source", "origin"):
            self.client.create_payload_index(self.collection, f, models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "ingested_at", models.PayloadSchemaType.INTEGER)

    def upsert(self, items: list[dict], wait: bool = True) -> None:
        pts = [models.PointStruct(
            id=point_id(i["payload"]["passage_id"]),
            vector={"dense": i["dense"], "bm25": models.SparseVector(indices=i["sp_idx"], values=i["sp_val"])},
            payload=i["payload"]) for i in items]
        self.client.upsert(self.collection, pts, wait=wait)

    def delete(self, passage_ids: list[str]) -> None:
        self.client.delete(self.collection, models.PointIdsList(points=[point_id(p) for p in passage_ids]), wait=True)

    def query_dense(self, vec, limit, flt=None):
        return self.client.query_points(self.collection, query=list(map(float, vec)), using="dense", limit=limit,
                                        query_filter=flt, with_payload=True).points

    def query_sparse(self, idx, val, limit, flt=None):
        if not idx:
            return []
        return self.client.query_points(self.collection, query=models.SparseVector(indices=idx, values=val),
                                        using="bm25", limit=limit, query_filter=flt, with_payload=True).points

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def wait_green(self, timeout: float = 900) -> str:
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.client.get_collection(self.collection).status
            if str(st).lower().endswith("green"):
                return "green"
            time.sleep(2)
        return "timeout"

    def info(self) -> dict:
        i = self.client.get_collection(self.collection)
        out = {"points": self.count(), "status": str(i.status).split(".")[-1].lower(),
               "indexed_vectors": getattr(i, "indexed_vectors_count", None), "categories": {}}
        try:
            out["categories"] = {h.value: h.count for h in
                                 self.client.facet(self.collection, key="category", limit=30).hits}
        except (ResponseHandlingException, UnexpectedResponse):
            pass
        return out
