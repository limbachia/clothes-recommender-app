from distutils.core import setup

setup(
    name='recommender_model',
    version='0.1.0',
    description='Clothes recommendations',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    # Substitute <github_account> with the name of your GitHub account
    url='https://github.com/<github_account>/<git repo>',
    author='Chirag B. Limbachia',  # Substitute your name
    author_email='chirag90in@yahoo.coin',  # Substitute your email
    license='MIT',
    packages=['recommender_model'],
    install_requires=[
        'pypandoc>=1.4',
        'pytest>=4.3.1',
        'pytest-runner>=4.4',
        'click>=7.0'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)