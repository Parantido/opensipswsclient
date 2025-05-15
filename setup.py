from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="opensipscall",  # Package name as it will appear on PyPI
    version="0.1.0",      # Package version (following semantic versioning)
    author="Your Name",   # The package author's name
    author_email="your.email@example.com",  # Contact email
    description="OpenSIPS WebSocket Client with ElevenLabs TTS Integration",  # Short description
    long_description=long_description,  # Full README content
    long_description_content_type="text/markdown",  # Format of long_description
    url="https://github.com/yourusername/opensipscall",  # Project homepage
    package_dir={"opensipscall": "lib"},  # Maps import namespace to directory
    packages=["opensipscall", "opensipscall.utils"],  # Packages to include
    classifiers=[  # Package metadata tags for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Minimum Python version required
    install_requires=[  # Required dependencies
        "websockets>=10.0",
        "pyaudio>=0.2.13",
        "elevenlabs>=0.2.26",
        "deepgram-sdk>=2.11.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={  # Optional dependencies
        "openai": ["openai>=1.0.0"],
        "groq": ["groq>=0.4.0"],
        "all": ["openai>=1.0.0", "groq>=0.4.0"]
    }
)
