from pathlib import Path
import os
class RootPath:
    rootpath = Path(__file__).resolve().parents[1]
def getAbsPath(PathInProject):
    return os.path.join(RootPath.rootpath,PathInProject)

