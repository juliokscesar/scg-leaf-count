import os
import shutil

def ensure_arg_in_kwargs(kwargs, *args_names):
    for arg in args_names:
        if arg not in kwargs:
            return False
        
    return True


def generate_temp_path(suffix: str) -> str:
    if not os.path.isdir(".temp"):
        os.mkdir(".temp")
    return os.path.join(".temp", os.urandom(24).hex()+suffix)


def clear_temp_folder():
    try:
        shutil.rmtree(".temp")
    except:
        print("DEBUG: couldn't delete temp images from '.temp'")


def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def get_all_files_from_paths(paths: list[str]):
    files = []
    for src in paths:
        if os.path.isfile(src):
            files.append(src)

        elif os.path.isdir(src):
            for (root, _, filenames) in os.walk(src):
                files.extend([os.path.join(root, file) for file in filenames])

        else:
            raise RuntimeError(f"{src} is an invalid image source")

    return files
