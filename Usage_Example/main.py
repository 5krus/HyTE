"""
Hyte Usage Example.

This script shows how to use HyTE with a practical example.
"""

# Prepare imports.
from hyte import Iterator            # LLM-based Hypothesize-Test-Evaluate iterator.
import tools                         # Functions, and their descriptions, for LLMs to use.
import controls                      # Prompts, tools, keys, and other consumables used by iterator.

# Run the iterator.
[summary, full_log] = Iterator().run(controls.SYSTEM_PROMPTS, # pylint: disable=not-callable.
                                     controls.OPTIONS,
                                     controls.KEY,
                                     tools.Tools(),
                                     sample_data = None)
