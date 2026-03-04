# Make the project root importable when running pytest from any directory.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
