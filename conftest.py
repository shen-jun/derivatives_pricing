"""
I keep this conftest.py at the project root purely so pytest adds the
project root to sys.path before collecting my tests. Every one of my test
modules imports from the `src` package using an absolute import (e.g.
`from src.models.black_scholes import BlackScholesModel`), and this is the
simplest way I found to make that work regardless of which directory
pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
