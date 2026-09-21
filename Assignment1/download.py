import gzip
from pathlib import Path
import urllib.request

RAW = Path(__file__).resolve().parent.parent / "data" / "FashionMNIST" / "raw"
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = RAW / name
        if path.exists():
            continue
        archive = RAW / (name + ".gz")
        if not archive.exists():
            print("Downloading %s" % archive.name)
            urllib.request.urlretrieve(BASE_URL + archive.name, archive)
        with gzip.open(archive, "rb") as source:
            path.write_bytes(source.read())
        print("Extracted %s" % path.name)

if __name__ == "__main__":
    main()
