from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='vectormorph',
    version='0.1.0',
    long_description=long_description,
    long_description_content_type="text/markdown", 
    packages=find_packages(),
    install_requires=[
        'hnswlib',
        'numpy'
    ],
    author='Collin Paran',
    description='A lightweight vector database alternative using HNSW and binary storage.',
)
