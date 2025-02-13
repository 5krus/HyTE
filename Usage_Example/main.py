"""
Hyte Usage Example.

This script shows how to use HyTE with a practical example.
"""

# Prepare imports.
import tools                         # Functions, and their descriptions, for LLMs to use.
import controls as ctl               # Prompts, tools, keys, and other consumables used by iterator.
from hyte import Iterator            # LLM-based Hypothesize-Test-Evaluate iterator.

# Run the iterator.
[summary, full_log] = Iterator(ctl.OPTIONS).run(ctl.SYSTEM_PROMPTS,  # pylint: disable=not-callable.
                                                tools.Tools(),
                                                sample_data = None)
