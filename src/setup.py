import setuptools
import os

loc = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Read the long description from the README file
with open(os.path.join(loc, 'README.md'), 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="action",
    version="0.0.1",
    description="Action",
    long_description=long_description,
    long_description_content_type="test/markdown",
    url="https://github.com/ekellbuch/action",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={},
    classifiers=["License :: OSI Approved :: MIT License"],
    python_requires=">=3.7",
)
