"""
Python conversion of the supplied C++ VectorDB + RAG HTTP server.

Requirements:
    Python 3.9+
    No third-party package is required.
    Ollama is optional and must be running locally for document/RAG endpoints.

Run:
    python vectordb_server.py

Server:
    http://localhost:8080
"""

import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DIMS = 16


# ---------------------------------------------------------------------
# DISTANCE METRICS
# ---------------------------------------------------------------------

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric):
    return {
        "cosine": cosine,
        "manhattan": manhattan,
        "euclidean": euclidean,
    }.get(metric, euclidean)


# ---------------------------------------------------------------------
# DATA TYPE
# ---------------------------------------------------------------------

class VectorItem:
    def __init__(self, item_id, metadata, category, emb):
        self.id = item_id
        self.metadata = metadata
        self.category = category
        self.emb = list(emb)


# ---------------------------------------------------------------------
# BRUTE FORCE INDEX
# ---------------------------------------------------------------------

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, item):
        self.items.append(item)

    def knn(self, query, k, dist):
        results = [(dist(query, item.emb), item.id) for item in self.items]
        return sorted(results)[:max(0, k)]

    def remove(self, item_id):
        self.items = [item for item in self.items if item.id != item_id]


# ---------------------------------------------------------------------
# KD-TREE INDEX
# ---------------------------------------------------------------------

class KDNode:
    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, dims):
        self.root = None
        self.dims = dims

    def _insert(self, node, item, depth):
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item):
        self.root = self._insert(self.root, item, 0)

    def _knn(self, node, query, k, depth, dist, heap):
        if node is None or k <= 0:
            return

        distance = dist(query, node.item.emb)
        entry = (-distance, -node.item.id)
        if len(heap) < k:
            import heapq
            heapq.heappush(heap, entry)
        elif distance < -heap[0][0]:
            import heapq
            heapq.heapreplace(heap, entry)

        axis = depth % self.dims
        diff = query[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, query, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, query, k, depth + 1, dist, heap)

    def knn(self, query, k, dist):
        import heapq
        heap = []
        self._knn(self.root, query, max(0, k), 0, dist, heap)
        return sorted([(-d, -item_id) for d, item_id in heap])

    def rebuild(self, items):
        self.root = None
        for item in items:
            self.insert(item)


# ---------------------------------------------------------------------
# HNSW INDEX
# ---------------------------------------------------------------------

class HNSW:
    class Node:
        def __init__(self, item, max_layer):
            self.item = item
            self.max_layer = max_layer
            self.nbrs = [[] for _ in range(max_layer + 1)]

    def __init__(self, m=16, ef_build=200):
        self.G = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)

    def _rand_level(self):
        # random() can theoretically return zero; protect log(0).
        u = max(self.rng.random(), 1e-12)
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, query, ep, ef, layer, dist):
        if ep not in self.G:
            return []
        visited = {ep}
        d0 = dist(query, self.G[ep].item.emb)
        candidates = [(d0, ep)]       # min-heap
        found = [(-d0, -ep)]          # max-heap represented by negatives
        import heapq
        heapq.heapify(candidates)
        heapq.heapify(found)

        while candidates:
            current_dist, current_id = heapq.heappop(candidates)
            worst_dist = -found[0][0]
            if len(found) >= ef and current_dist > worst_dist:
                break
            node = self.G.get(current_id)
            if node is None or layer >= len(node.nbrs):
                continue

            for neighbor_id in node.nbrs[layer]:
                if neighbor_id in visited or neighbor_id not in self.G:
                    continue
                visited.add(neighbor_id)
                neighbor_dist = dist(query, self.G[neighbor_id].item.emb)
                if len(found) < ef or neighbor_dist < -found[0][0]:
                    heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                    heapq.heappush(found, (-neighbor_dist, -neighbor_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted([(-d, -item_id) for d, item_id in found])

    @staticmethod
    def _select_neighbors(candidates, max_m):
        return [item_id for _, item_id in candidates[:max_m]]

    def insert(self, item, dist):
        item_id = item.id
        level = self._rand_level()
        self.G[item_id] = self.Node(item, level)

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = level
            return

        ep = self.entry_pt
        for layer in range(self.top_layer, level, -1):
            if layer < len(self.G[ep].nbrs):
                candidates = self._search_layer(item.emb, ep, 1, layer, dist)
                if candidates:
                    ep = candidates[0][1]

        for layer in range(min(self.top_layer, level), -1, -1):
            candidates = self._search_layer(item.emb, ep, self.ef_build, layer, dist)
            max_m = self.M0 if layer == 0 else self.M
            selected = self._select_neighbors(candidates, max_m)
            self.G[item_id].nbrs[layer] = selected

            for neighbor_id in selected:
                if neighbor_id not in self.G:
                    continue
                neighbor = self.G[neighbor_id]
                while len(neighbor.nbrs) <= layer:
                    neighbor.nbrs.append([])
                connections = neighbor.nbrs[layer]
                connections.append(item_id)

                if len(connections) > max_m:
                    distances = [
                        (dist(neighbor.item.emb, self.G[c].item.emb), c)
                        for c in connections if c in self.G
                    ]
                    distances.sort()
                    neighbor.nbrs[layer] = [c for _, c in distances[:max_m]]

            if candidates:
                ep = candidates[0][1]

        if level > self.top_layer:
            self.top_layer = level
            self.entry_pt = item_id

    def knn(self, query, k, ef, dist):
        if self.entry_pt == -1 or k <= 0:
            return []
        ep = self.entry_pt

        for layer in range(self.top_layer, 0, -1):
            if layer < len(self.G[ep].nbrs):
                candidates = self._search_layer(query, ep, 1, layer, dist)
                if candidates:
                    ep = candidates[0][1]

        results = self._search_layer(query, ep, max(ef, k), 0, dist)
        return results[:k]

    def remove(self, item_id):
        if item_id not in self.G:
            return
        for node in self.G.values():
            for layer in node.nbrs:
                layer[:] = [neighbor for neighbor in layer if neighbor != item_id]
        if self.entry_pt == item_id:
            remaining = [node_id for node_id in self.G if node_id != item_id]
            self.entry_pt = remaining[0] if remaining else -1
            if self.entry_pt == -1:
                self.top_layer = -1
            else:
                self.top_layer = max(node.max_layer for node in self.G.values()
                                     if node.item.id != item_id)
        del self.G[item_id]

    def get_info(self):
        max_layers = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_layers
        edges_per_layer = [0] * max_layers
        nodes = []
        edges = []

        for item_id, node in self.G.items():
            nodes.append({
                "id": item_id,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_layer,
            })
            for layer in range(min(node.max_layer + 1, max_layers)):
                nodes_per_layer[layer] += 1
                if layer < len(node.nbrs):
                    for neighbor_id in node.nbrs[layer]:
                        if item_id < neighbor_id:
                            edges_per_layer[layer] += 1
                            edges.append({
                                "src": item_id, "dst": neighbor_id, "lyr": layer
                            })

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def __len__(self):
        return len(self.G)


# ---------------------------------------------------------------------
# VECTOR DATABASE
# ---------------------------------------------------------------------

class VectorDB:
    def __init__(self, dims):
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.RLock()
        self.next_id = 1
        self.dims = dims

    def insert(self, metadata, category, emb, dist):
        with self.lock:
            item = VectorItem(self.next_id, metadata, category, emb)
            self.next_id += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, query, k, metric, algo):
        with self.lock:
            dist = get_dist_fn(metric)
            start = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self.bf.knn(query, k, dist)
            elif algo == "kdtree":
                raw = self.kdt.knn(query, k, dist)
            else:
                raw = self.hnsw.knn(query, k, 50, dist)
            elapsed_us = (time.perf_counter_ns() - start) // 1000

            hits = []
            for distance, item_id in raw:
                item = self.store.get(item_id)
                if item:
                    hits.append({
                        "id": item.id,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": round(distance, 6),
                        "embedding": item.emb,
                    })
            return {
                "results": hits,
                "latencyUs": elapsed_us,
                "algo": algo,
                "metric": metric,
            }

    def benchmark(self, query, k, metric):
        with self.lock:
            dist = get_dist_fn(metric)

            def measure(fn):
                start = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - start) // 1000

            return {
                "bruteforceUs": measure(lambda: self.bf.knn(query, k, dist)),
                "kdtreeUs": measure(lambda: self.kdt.knn(query, k, dist)),
                "hnswUs": measure(lambda: self.hnsw.knn(query, k, 50, dist)),
                "itemCount": len(self.store),
            }

    def all(self):
        with self.lock:
            return [
                {
                    "id": item.id,
                    "metadata": item.metadata,
                    "category": item.category,
                    "embedding": item.emb,
                }
                for item in self.store.values()
            ]

    def hnsw_info(self):
        with self.lock:
            return self.hnsw.get_info()

    def size(self):
        with self.lock:
            return len(self.store)


# ---------------------------------------------------------------------
# TEXT CHUNKER AND OLLAMA CLIENT
# ---------------------------------------------------------------------

def chunk_text(text, chunk_words=250, overlap_words=30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    step = chunk_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])
        chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


class OllamaClient:
    def __init__(self, host="127.0.0.1", port=11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def _post(self, path, payload, timeout):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def is_available(self):
        try:
            request = urllib.request.Request(self.base_url + "/api/tags")
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def embed(self, text):
        try:
            # Keep compatibility with the C++ program's /api/embeddings endpoint.
            result = self._post(
                "/api/embeddings",
                {"model": self.embed_model, "prompt": text},
                30,
            )
            return result.get("embedding", [])
        except (OSError, urllib.error.URLError, ValueError, KeyError):
            return []

    def generate(self, prompt):
        try:
            result = self._post(
                "/api/generate",
                {"model": self.gen_model, "prompt": prompt, "stream": False},
                180,
            )
            return result.get("response", "")
        except (OSError, urllib.error.URLError, ValueError):
            return "ERROR: Ollama unavailable. Run: ollama serve"


# ---------------------------------------------------------------------
# DOCUMENT DATABASE / RAG
# ---------------------------------------------------------------------

class DocItem:
    def __init__(self, item_id, title, text, emb):
        self.id = item_id
        self.title = title
        self.text = text
        self.emb = list(emb)


class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = threading.RLock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title, text, emb):
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            elif len(emb) != self.dims:
                raise ValueError("Embedding dimensions do not match existing documents")

            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vector_item = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vector_item, cosine)
            self.bf.insert(vector_item)
            return item.id

    def search(self, query, k, max_dist=0.7):
        with self.lock:
            if not self.store:
                return []
            raw = (self.bf.knn(query, k, cosine) if len(self.store) < 10
                   else self.hnsw.knn(query, k, 50, cosine))
            output = []
            for distance, item_id in raw:
                item = self.store.get(item_id)
                if item is not None and distance <= max_dist:
                    output.append((distance, item))
            return output

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self):
        with self.lock:
            result = []
            for item in self.store.values():
                preview = item.text[:120]
                if len(item.text) > 120:
                    preview += "…"
                result.append({
                    "id": item.id,
                    "title": item.title,
                    "preview": preview,
                    "words": len(item.text.split()),
                })
            return result

    def size(self):
        with self.lock:
            return len(self.store)


# ---------------------------------------------------------------------
# DEMO DATA
# ---------------------------------------------------------------------

def load_demo(db):
    dist = cosine
    demo = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for metadata, category, embedding in demo:
        db.insert(metadata, category, embedding, dist)


# ---------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()
load_demo(db)


class APIHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, body, content_type="text/plain; charset=utf-8", status=200):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _body_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        def param(name, default=""):
            return params.get(name, [default])[0]

        if path == "/":
            try:
                with open("index.html", "r", encoding="utf-8") as file:
                    self._send_text(file.read(), "text/html; charset=utf-8")
            except OSError:
                self._send_text("index.html not found", status=404)

        elif path == "/search":
            query = parse_vector(param("v"))
            if len(query) != DIMS:
                self._send_json({"error": f"need {DIMS}D vector"})
                return
            try:
                k = int(param("k", "5"))
            except ValueError:
                k = 5
            metric = param("metric", "cosine") or "cosine"
            algo = param("algo", "hnsw") or "hnsw"
            self._send_json(db.search(query, k, metric, algo))

        elif path == "/items":
            self._send_json(db.all())

        elif path == "/benchmark":
            query = parse_vector(param("v"))
            if len(query) != DIMS:
                self._send_json({"error": f"need {DIMS}D vector"})
                return
            try:
                k = int(param("k", "5"))
            except ValueError:
                k = 5
            metric = param("metric", "cosine") or "cosine"
            self._send_json(db.benchmark(query, k, metric))

        elif path == "/hnsw-info":
            self._send_json(db.hnsw_info())

        elif path == "/doc/list":
            self._send_json(doc_db.all())

        elif path == "/status":
            self._send_json({
                "ollamaAvailable": ollama.is_available(),
                "embedModel": ollama.embed_model,
                "genModel": ollama.gen_model,
                "docCount": doc_db.size(),
                "docDims": doc_db.dims,
                "demoDims": DIMS,
                "demoCount": db.size(),
            })

        elif path == "/stats":
            self._send_json({
                "count": db.size(),
                "dims": DIMS,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"],
            })

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body_json()

        if path == "/insert":
            metadata = body.get("metadata", "")
            category = body.get("category", "")
            embedding = body.get("embedding", [])
            if not metadata or not isinstance(embedding, list) or len(embedding) != DIMS:
                self._send_json({"error": "invalid body"})
                return
            try:
                embedding = [float(value) for value in embedding]
                item_id = db.insert(metadata, category, embedding, cosine)
                self._send_json({"id": item_id})
            except (TypeError, ValueError):
                self._send_json({"error": "invalid body"})

        elif path == "/doc/insert":
            title = body.get("title", "")
            text_value = body.get("text", "")
            if not title or not text_value:
                self._send_json({"error": "need title and text"})
                return

            chunks = chunk_text(text_value, 250, 30)
            ids = []
            for index, chunk in enumerate(chunks):
                embedding = ollama.embed(chunk)
                if not embedding:
                    self._send_json({
                        "error": "Ollama unavailable. Install from https://ollama.com "
                                 "then run: ollama pull nomic-embed-text && "
                                 "ollama pull llama3.2"
                    })
                    return
                chunk_title = (
                    f"{title} [{index + 1}/{len(chunks)}]"
                    if len(chunks) > 1 else title
                )
                try:
                    ids.append(doc_db.insert(chunk_title, chunk, embedding))
                except ValueError as error:
                    self._send_json({"error": str(error)})
                    return

            self._send_json({
                "ids": ids,
                "chunks": len(chunks),
                "dims": doc_db.dims,
            })

        elif path == "/doc/search":
            question = body.get("question", "")
            try:
                k = int(body.get("k", 3))
            except (TypeError, ValueError):
                k = 3
            if not question:
                self._send_json({"error": "need question"})
                return
            query_embedding = ollama.embed(question)
            if not query_embedding:
                self._send_json({"error": "Ollama unavailable"})
                return
            hits = doc_db.search(query_embedding, k)
            self._send_json({
                "contexts": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "distance": round(distance, 4),
                    }
                    for distance, item in hits
                ]
            })

        elif path == "/doc/ask":
            question = body.get("question", "")
            try:
                k = int(body.get("k", 3))
            except (TypeError, ValueError):
                k = 3
            if not question:
                self._send_json({"error": "need question"})
                return

            query_embedding = ollama.embed(question)
            if not query_embedding:
                self._send_json({"error": "Ollama unavailable"})
                return

            hits = doc_db.search(query_embedding, k)
            context_parts = []
            for index, (distance, item) in enumerate(hits, 1):
                context_parts.append(f"[{index}] {item.title}:\n{item.text}\n")
            context = "\n".join(context_parts)

            prompt = (
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it doesn't, just use your own general knowledge. "
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say "
                "things like 'the context doesn't mention'. Just answer the question "
                "naturally.\n\n"
                f"Context:\n{context}"
                f"Question: {question}\n\nAnswer:"
            )
            answer = ollama.generate(prompt)
            self._send_json({
                "answer": answer,
                "model": ollama.gen_model,
                "contexts": [
                    {
                        "id": item.id,
                        "title": item.title,
                        "text": item.text,
                        "distance": round(distance, 4),
                    }
                    for distance, item in hits
                ],
                "docCount": doc_db.size(),
            })

        else:
            self._send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path

        if path.startswith("/delete/"):
            try:
                item_id = int(path.rsplit("/", 1)[1])
            except ValueError:
                self._send_json({"error": "invalid id"}, 400)
                return
            self._send_json({"ok": db.remove(item_id)})

        elif path.startswith("/doc/delete/"):
            try:
                item_id = int(path.rsplit("/", 1)[1])
            except ValueError:
                self._send_json({"error": "invalid id"}, 400)
                return
            self._send_json({"ok": doc_db.remove(item_id)})

        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        # Keep the terminal output simple, similar to the original server.
        print("%s - %s" % (self.address_string(), fmt % args))


def parse_vector(value):
    if not value:
        return []
    try:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError:
        return []


def main():
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama: " + ("ONLINE" if ollama_up else
                       "OFFLINE (install from ollama.com)"))
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    server = ThreadingHTTPServer(("0.0.0.0", 8080), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VectorDB server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
