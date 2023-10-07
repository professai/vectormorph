from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='vectormorph',
    version='0.1.1',
    author='Collin Paran',
    description='A lightweight vector database.',
    long_description=long_description,
    long_description_content_type="text/markdown", 
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'hnswlib',
        'uvicorn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache-2.0', 
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'serve=vectormorph.vector_morph:main',
        ],
    },
)
