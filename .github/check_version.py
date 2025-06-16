import os
import tomllib  # Python 3.11+

TAG = os.environ.get("TAG_NAME")
if TAG and TAG.startswith("v"):
    TAG = TAG[1:]

with open("go3/pyproject.toml", "rb") as f:
    data = tomllib.load(f)

version = data["project"]["version"]

if version != TAG:
    raise SystemExit(f"❌ Version mismatch: pyproject.toml={version} vs tag={TAG}")
else:
    print(f"✅ Version match: {version}")