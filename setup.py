import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlflow_redisai",
    version="0.0.1",
    author="hhsecond",
    author_email="sherin@tensorwerk.com",
    description="MLFlow RedisAI integration package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hhsecond/mlflow-redisai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'mlflow_redisai=mlflow_redisai.cli:main',
        ]}
)
