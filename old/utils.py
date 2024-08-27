import os
import shutil
import yaml

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


def get_all_files_from_paths(*args):
    files = []
    for path in args:
        if os.path.isfile(path):
            files.append(path)

        elif os.path.isdir(path):
            for (root, _, filenames) in os.walk(path):
                files.extend([os.path.join(root, file) for file in filenames])

        else:
            raise RuntimeError(f"{path} is an invalid image source")

    return files


def read_yaml(yaml_file: str):
    content = {}
    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    return content

