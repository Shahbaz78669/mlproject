from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path) as file_obj:
        requirement = file_obj.readlines()

        for req in requirement:
            req = req.replace('\n', '')

            if req != "-e .":
                requirements.append(req)

    return requirements


setup(
    author="Shahbaz Khan",
    author_email="shahbazkhan211016@gmail.com",
    name="myproject",
    version="1.0.0",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)