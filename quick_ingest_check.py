import json
import os
import tempfile

from evidence.session_store import SessionWorkspace

root = tempfile.mkdtemp()
ws = SessionWorkspace(root)
h = ws.create_session("Building A")
d = tempfile.mkdtemp()
p = os.path.join(d, "a.jpg")
with open(p, "wb") as f:
    f.write(b"\xff\xd8\xff\xe0" + os.urandom(64))
r = h.add_source(p)
ir = h.ingest_source(r.source_id)
h2 = ws.open_session("Building-A")
print("counts:", len(h2.session.all_evidence()), len(h2.package.all_assets()), len(h2.sources))
