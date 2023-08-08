import setuptools
from setuptools import setup, find_packages
from torchxrayvision import _version

with open("README.md", "r") as fh:
    long_description = fh.read()
    
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setuptools.setup(
    name="torchxrayvision",
    version=_version.__version__,
    author="Artin Majdi",
    author_email="msm2024@gmail.com",
    description="Taxonomy Based Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlmed/torchxrayvision",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires='>=3.7',
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    package_dir={'taxonomy': 'taxonomy'},
    package_data={'taxonomy': ['taxonomy/config.json']},
    include_package_data=True,
    zip_safe=False,
    entry_points={'console_scripts': ['taxonomy = taxonomy.taxonomy:main',]},
)
