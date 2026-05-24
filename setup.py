
from setuptools import setup, find_packages

setup(
    name="sign-language-recognition",
    version="1.0.0",
    description="Production-ready Sign Language Recognition using MediaPipe, CNN, and Transformer",
    author="Ankith Singh",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    include_package_data=True,
)
