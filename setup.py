from distutils.core import setup, find_packages
setup(
    name = 'nativefields',
    packages=find_packages(exclude=['tests*']),
    version = '1.0.0',
    license='GPL3',
    description = 'Allows for easy interpreting of bytearrays as native class fields',
    author = 'Adam Bieńkowski',
    author_email = 'donadigos159@gmail.com',
    url = 'https://github.com/donadigo/nativefields',
    download_url = 'https://github.com/donadigo/TMInterfaceClientPython/archive/refs/tags/0.1.tar.gz',
    keywords = ['struct', 'nativefields', 'bytearray'],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)