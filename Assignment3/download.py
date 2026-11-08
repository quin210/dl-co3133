from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceM4/A-OKVQA"
# revision named in the M1 proposal (short sha d1b0efa)
REVISION = "d1b0efa3a436e9101dfbde3752db7607da696c35"
ROOT = Path(__file__).resolve().parent.parent / "data" / "aokvqa"

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
    for name in sorted(files):
        if not name.endswith(".parquet"):
            continue
        if (ROOT / name).exists():
            print("have", name)
            continue
        # the test parquet is only used to count rows, it has no labels
        print("downloading", name)
        hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=ROOT)

if __name__ == "__main__":
    main()
