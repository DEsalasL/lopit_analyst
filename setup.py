from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='lopit_analyst',
    version='1.0.0',
    author='Dayana E. Salas-Leiva',
    author_email='ds2000@cam.ac.uk',
    description='A program for analysing TMT labeled proteomics data that '
                'have been pre-processed with Proteome Discoverer 3.1+ '
                '(Thermo Scientific)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=([
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]),
    scripts=['bin/charms.py', 'bin/clustering_data.py', 'bin/data_filering.py',
             'bin/graphic_results.py', 'bin/lopit_analyst.py',
             'bin/lopit_menu.py', 'bin/lopit_utils.py',
             'bin/mv_imp_norm_aggr.py', 'bin/mv_removal.py',
             'bin/psm_diagnostics.py', 'bin/svm_knn_rf_clustering.py'],
    entry_points={
        'console_scripts': [
            'lopit_analyst = lopit_analyst:main',
        ],
    }
)