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

