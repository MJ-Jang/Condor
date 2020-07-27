from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='condor',
    version='0.1',
    description='Korean spacing corrector',
    author='MJ Jang',
    install_requires=required,
    packages=find_packages(exclude=['docs', 'tests', 'tmp', 'data', '__pycache__']),
    python_requires='>=3',
    package_data={'condor': ['resources/*']},
    include_package_data=True,
    zip_safe=False,
)
