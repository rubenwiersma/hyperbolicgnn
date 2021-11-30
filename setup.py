import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='hgnn_replication',
    version='0.1',
    author='Ruben Wiersma, Metehan Doyran',
    author_email="rubenwiersma@gmail.com, m.doyran@uu.nl",
    description='Replication of Hyperbolic GNNs synthetic dataset experiment',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rubenwiersma/hyperbolicgnn",
    project_urls={
        "Bug Tracker": "https://github.com/rubenwiersma/hyperbolicgnn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where=".", include=["hgnn", "hgnn.*"]),
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'pytorch',
        'pyg',
        'argparse',
        'yaml',
        'networkx',
        'easydict',
        'tensorboard'
    ],
)
