from setuptools import setup

setup(
    name='tts_metric_tools',
    version='0.1',
    description='Tools for computing metrics between speech samples.',
    url='https://github.com/ZackHodari/tts_metric_tools',
    author='Zack Hodari',
    author_email='zack.hodari@ed.ac.uk',
    # license='MIT',
    packages=['tts_metric_tools'],
    entry_points={'console_scripts': ['tmt_process = tts_metric_tools.process:main',
                                      'tmt_metrics = tts_metric_tools.metrics:main']})
