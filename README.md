![HyTE Logo](https://i.imgur.com/VuzljTm.gif)

## Description:
Tooling for running LLM-based Hypothesize-Test-Evaluate methodology.

#### Process Diagram:

![HyTE Methodology Diagram](https://i.imgur.com/i2e2Z0k.png)

i.e. Models creates a hypothesis, test it experimentally, and evalaute whether the results support the original idea. This new information is fed back into the Hypothesis LLM such that it has further context for its next hypothesis. This loop occurs `ITERATION` number of times. This methodology and tooling is (mostly) field, experiment and model agnostic.

## Installation:
The package is available through PyPI; it is installed with the following command:
```
pip install HyTE
```

## Usage:

Primary usage only requires one command, `Iterator.run(...)`. i.e.

```
# Prepare imports.
from HyTE import *          # LLM-based Hypothesize-Test-Evaluate iterator.

# Run the iterator.
[summary, full_log] = Iterator().run(SYSTEM_PROMPTS, OPTIONS, KEY, Tools(), sample_data = None)
```
A more comprehensive, practical usage example is provided [here](https://github.com/5krus/HyTE/tree/prod/Usage_Example).

## Credit:
Primary development completed and maintained by [Eryk Krusinski](https://www.eng.cam.ac.uk/profiles/ek620), with feedback from [Dr Viacheslav Sedunin](https://www.eng.cam.ac.uk/profiles/vs440) and [Dr James V. Taylor](https://www.eng.cam.ac.uk/profiles/jvt24). Correspondence: [ek620@cam.ac.uk](ek620@cam.ac.uk).
Research funding for project during which this idea was conceived provided by [InnovateUK](https://www.ukri.org/councils/innovate-uk/).