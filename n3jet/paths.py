import logging
import os
try:
    from pathlib2 import Path
except:
    from pathlib import Path
from sys import argv

logger = logging.getLogger(
    __name__
)

project_directory = Path(
    os.path.abspath(__file__)
).parent.parent

working_directory = Path(
    os.getcwd()
)

working_directory_parent = working_directory.parent


def find_default(name):
    """
    Get a default path when no command line argument is passed.
    - First attempt to find the folder in the current working directory.
    - If it is not found there then try the directory in which June lives.
    - Finally, try the directory above the current working directory. This
    is for the build pipeline.
    This means that tests will find the configuration regardless of whether
    they are run together or individually.
    Parameters
    ----------
    name
        The name of some folder
    Returns
    -------
    The full path to that directory
    """
    for directory in (
        working_directory,
        project_directory,
        working_directory_parent
    ):
        path = directory / name
        if os.path.exists(
                path
        ):
            return path
    raise FileNotFoundError(
        "Could not find a default path for {}".format(name)
    )


def path_for_name(name):
    """
    Get a path input using a flag when the program is run.
    If no such argument is given default to the directory above
    the june with the name of the flag appended.
    e.g. --data indicates where the data folder is and defaults
    to june/../data
    Parameters
    ----------
    name
        A string such as "data" which corresponds to the flag --data
    Returns
    -------
    A path
    """
    flag = "--{}".format(name)
    try:
        path = Path(argv[argv.index(flag) + 1])
        if not path.exists():
            raise FileNotFoundError(
                "No such folder {}".format(path)
            )
    except (IndexError, ValueError):
        path = find_default(name)
        logger.warning(
            "No {} argument given - defaulting to:\n{}".format(flag, path)
        )

    return path


configs_path = path_for_name("configs")
