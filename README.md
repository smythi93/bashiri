# BASHIRI: Learning Failure Oracles from Test Suites

BASHIRI is an approach to infer failure oracles from a small test suite automatically.

## Abstract

Program fixes must preserve passing tests while fixing failing ones. Validating these properties requires _test oracles_ that distinguish passing from failing runs.
	
We introduce _BASHIRI_, an approach that learns _failure oracles_ from test suites with labeled outcomes using execution features. _BASHIRI_ leverages execution-feature-driven debugging to collect program execution features and trains interpretable models as testing oracles.

Our evaluation shows that _BASHIRI_ predicts test outcomes with 95% accuracy, effectively identifying failing runs.
	
_BASHIRI_ is available as an open-source tool at
https://github.com/smythi93/bashiri

A demonstration video is available at
https://youtu.be/D2mJkCtSXtM

## Installation

To install BASHIRI and its dependencies, run the following command:
```bash
python -m pip install .
```

## Usage

For BASHIRI, you need to instrument your subject.
```python
instrument("middle.py", "tmp.py", "middle.json")
```
Next, you need some tests to execute and collect their event traces. 
We provide two collectors, one for unit tests and one for input to the program.
However, implementing another collector by inheriting the base class `EventCollector` and implementing its `collect()` method is an option.
To employ the collector, use it like this:
```python
oracle = Bashiri(
    "middle.py",
    (passing, failing),
    access="tmp.py",
    mapping="middle.json",
    work_dir=".",
)
oracle.learn()
``` 

Now, we can leverage the collector, handle, and oracle to identify the result of an unseen execution/test case with high accuracy.
```python
result = oracle.predict(unseen_test)
```

## Mapping

The mapping approach infers connections between the events if different versions of a program, e.g., a faulty and a fixed version.
This mapping allows to continuously apply the learned oracles.

```python
patch = PatchTranslator.build_t4p_translator(project)
creator = MappingCreator(mapping_bug)
mapping = creator.create(mapping_fix, patch)
translation_mapping = EventMapping(
    mapping_bug.mapping,
    translation=mapping.get_translation(),
    alternative_mapping=mapping_fix.mapping,
)
```

## Human-In-The-Loop

We have additionally implemented a mechanism to incorporate a human oracle to enrich insufficient test sets in the `HumanOracleRefinement` class.
```python
seeds = {i: s, for i, s in enumerate(passing + failing)}
refinement = HumanOracleRefinement(oracle, seeds, collector)
refinement.run()
```

# Example

We have provided a short example in the `example.ipynb` notebook.
This notebook demonstrates how to use BASHIRI with a simple example, including instrumentation, test collection, and oracle learning.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
