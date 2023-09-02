import setuptools
from setuptools import find_packages

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setuptools.setup(
	name="taxonomy",
	version="1.0.0",
	author="Artin Majdi",
	author_email="msm2024@gmail.com",
	description="Taxonomy Based Classification",
	url="https://github.com/artinmajdi/taxonomy",
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
	entry_points={'console_scripts': ['taxonomy = taxonomy.taxonomy:main', ]},
)
