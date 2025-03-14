from distutils.core import setup

setup(
    name='evclust',
    version='0.2.1',
    author='Armel SOUBEIGA',
    author_email='armel.soubeiga@yahoo.fr',
    packages=['evclust','evclust.test'],
    scripts=[],
    url='http://pypi.python.org/pypi/evclust/',
    license='LICENSE',
    description='Evidential Clustering',
    long_description=open('README.txt').read(),
        install_requires=[
        "caldav == 0.1.4",
    ],
)
