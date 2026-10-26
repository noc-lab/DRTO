from setuptools import find_packages, setup

setup(
    name="drto",
    version="0.1.0",
    description="Distributionally Robust Token Optimization (DRTO) with span-level DRO",
    packages=find_packages(include=["openrlhf", "openrlhf.*"]),
    python_requires=">=3.10",
    license="Apache-2.0",
)
