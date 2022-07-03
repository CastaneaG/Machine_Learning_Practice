from pathlib import Path
import os
class RootPath:
    rootpath = Path(__file__).resolve().parents[1]
def getAbsPath(PathInProject):
    """

    :param PathInProject: 文件在项目中路径
    :return: 自动拼接的项目绝对路径+文件在项目中路径
    """
    return os.path.join(RootPath.rootpath,PathInProject)

