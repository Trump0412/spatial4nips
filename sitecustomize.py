import re
import importlib
import yaml

def _resolve_task_dir_from_loader(loader):
    # 试图从 loader 的 stream 文件路径推断任务目录名（vsibench / MMSI_Video_Bench / ...）
    stream = getattr(loader, "stream", None)
    path = getattr(stream, "name", None)
    if not path:
        return None
    path = path.replace("\\", "/")
    m = re.search(r"/lmms_eval/tasks/([^/]+)/", path)
    return m.group(1) if m else None

def _function_constructor(loader, node):
    """
    Parse: !function utils.some_func
    -> import lmms_eval.tasks.<taskdir>.utils, return getattr(module, some_func)
    """
    value = loader.construct_scalar(node)  # e.g. "utils.MMSI_Video_Bench_doc_to_visual"
    if "." not in value:
        raise ValueError(f"Bad !function spec: {value}")
    mod, attr = value.rsplit(".", 1)

    if mod == "utils":
        taskdir = _resolve_task_dir_from_loader(loader)
        if taskdir:
            mod = f"lmms_eval.tasks.{taskdir}.utils"

    module = importlib.import_module(mod)
    fn = getattr(module, attr)
    return fn

# Register for common loaders
for L in (yaml.SafeLoader, yaml.FullLoader):
    try:
        L.add_constructor("!function", _function_constructor)
    except Exception:
        pass
