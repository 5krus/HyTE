# Prepare imports.
import HyTE                 # LLM-based Hypothesize-Test-Evaluate iterator.
from tools import *         # Functions, and their descriptions, for LLMs to use.
from controls import *      # Prompts, tools, keys, and other consumables used by the iterator.

# Run the iterator.
[summary, full_log] = HyTE.run(SYSTEM_PROMPTS, OPTIONS, KEY, Tools(), sample_data = None)