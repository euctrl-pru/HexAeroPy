# setup.py
from setuptools import setup, find_packages

setup(
    name='HexAeroPy',
    version='1.0.2',
    packages=find_packages(),
    include_package_data=True,  
    package_data={
        'HexAeroPy': ['data/*', 'data/*/*', 'data/*/*/*'],  
    },
    install_requires=[
        'h3>=3.7.6',
        'pandas>=1.5.2',
        'matplotlib>=3.6.3',
        'geojson>=3.1.0',
        'folium>=0.14.0',
        'requests>=2.27.0',
        'tqdm>=4.0.0', 
        'h3pandas>=0.2.6'
    ],
    author='Quinten Goens',
    author_email='quinten.goens@eurocontrol.int',
    description='A EUROCONTROL package to determine used airports, runways, taxiways and stands based on flight trajectory coordinates.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/euctrl-pru/hexaeropy',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ] 
)
