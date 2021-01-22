import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='setigen',
    version='1.4.0',
    author='Bryan Brzycki',
    author_email='bbrzycki@berkeley.edu',
    description='SETI radio signal generator',
    long_description='Python library for generating and injecting artificial narrow-band signals into time-frequency data',
    long_description_content_type='text/markdown',
    url='https://github.com/bbrzycki/setigen',
    project_urls={
        'Documentation': 'https://setigen.readthedocs.io/en/latest/',
        'Source': 'https://github.com/bbrzycki/setigen'
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
       'numpy>=1.18.1',
       'scipy>=1.4.1',
       'astropy>=4.0',
       'blimpy>=2.0.0',
       'matplotlib>=3.1.3',
       'sphinx-rtd-theme==0.4.3'
    ],
    dependency_links=['https://github.com/h5py/h5py',
                      'https://github.com/kiyo-masui/bitshuffle'],
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
)
