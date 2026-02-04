import setuptools

version_file = "src/athutils/__version__.py"

def get_version():

    with open(version_file, 'r') as file:
        _line = file.read()

    __version__ = _line.split("=")[-1].lstrip(" '").rstrip(" '\n")

    return __version__

PACKAGE_DATA = {}

# Thank you:
setuptools.setup(
    version = get_version(),
    package_data=PACKAGE_DATA,
)
