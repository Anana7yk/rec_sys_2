import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for relative in ["recsys","experiments/study03","experiments/study04"]:
 sys.path.insert(0,str(ROOT/relative))
