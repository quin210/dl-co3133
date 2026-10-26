from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.parse
import urllib.request

from huggingface_hub import HfApi

REPO = "shreemunale/tomato-leaves-dataset"
# the revision named in the M1 proposal
REVISION = "c029791b678863f6cb856589e4eea8d8f78e5931"
RAW = Path(__file__).resolve().parent.parent / "data" / "tomato" / "raw"
PREFIX = "dataset-tomatoes/"

def local_path(remote):
    # some source file names contain "?" which Windows cannot store
    return RAW / remote[len(PREFIX):].replace("?", "_")

def fetch(remote):
    path = local_path(remote)
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/datasets/%s/resolve/%s/%s" % (REPO, REVISION, urllib.parse.quote(remote))
    for attempt in range(8):
        try:
            # write beside the target so an interrupted download is never mistaken for a finished one
            part = path.with_name(path.name + ".part")
            urllib.request.urlretrieve(url, part)
            part.replace(path)
            return
        except Exception:
            if attempt == 7:
                raise
            # back off when the server answers 429 Too Many Requests
            time.sleep(2 ** attempt)

def main():
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION) if f.startswith(PREFIX)]
    print("Downloading %d files" % len(files))
    with ThreadPoolExecutor(12) as pool:
        for done, _ in enumerate(pool.map(fetch, files), 1):
            if done % 2000 == 0:
                print("%d / %d" % (done, len(files)), flush=True)
    print("Done: %s" % RAW)

if __name__ == "__main__":
    main()
