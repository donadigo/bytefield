from setuptools import setup, find_packages

setup(
    name="bytefields",
    packages=find_packages(exclude=["tests*"]),
    version="1.0.0",
    license="GPL3",
    description="Parse binary data using declarative field layout and native Python properties",
    author="Adam Bieńkowski",
    author_email="donadigos159@gmail.com",
    url="https://github.com/donadigo/bytefields",
    download_url="",
    keywords=["struct", "bytefields", "bytearray"],
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
