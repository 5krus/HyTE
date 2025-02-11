![HyTE Logo](https://i.imgur.com/VuzljTm.gif)

## Description:
Automation of LLM-based Hypothesize-Test-Evaluate methodology.
<br/><br/>

#### Process Diagram:
<img src="https://i.imgur.com/i2e2Z0k.png" width=40%>

Models create a hypothesis, test it experimentally, and evalaute whether the results support their original idea. This new information is fed back into the Hypothesis LLM, giving it further context for its next hypothesis. This loop occurs `ITERATION` number of times. <br/><br/>This methodology and tooling is (mostly) field, experiment and model agnostic.

## Installation:
The package is available through [PyPI](https://pypi.org/project/HyTE/0.0.1/).<br/>It is installed with the following command:
```
pip install HyTE
```

## Usage:

Primary usage only requires one command, `Iterator.run(...)`. i.e.

```
# Prepare imports.
from HyTE import *    # Obtain LLM-based Hypothesize-Test-Evaluate iterator.

# Run the iterator.
[summary, full_log] = Iterator().run(SYSTEM_PROMPTS, OPTIONS, KEY, Tools())
```
A more comprehensive, practical usage example is provided [here](https://github.com/5krus/HyTE/tree/prod/Usage_Example).

## Credit:
Primary development completed and maintained by [Eryk Krusinski](https://www.eng.cam.ac.uk/profiles/ek620), with feedback from [Dr Viacheslav Sedunin](https://www.eng.cam.ac.uk/profiles/vs440) and [Dr James V. Taylor](https://www.eng.cam.ac.uk/profiles/jvt24). Correspondence: [ek620@cam.ac.uk](ek620@cam.ac.uk). <br/>
[InnovateUK](https://www.ukri.org/councils/innovate-uk/) funded project during which this idea was conceived.
