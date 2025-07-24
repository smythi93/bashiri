# BASHIRI: Learning Failure Oracles from Test Suites

BASHIRI is an approach to infer failure oracles from a small test suite automatically.

## Abstract

When fixing a program-be it manually or through automated repair-it is crucial that the fix (1) entirely fixes previously failing runs and (2) does not impact previously passing runs.
Both properties are typically validated by a _test suite_ that leverages _oracles_ to determine if a run is passing or failing.
   
We introduce _BASHIRI_, an approach that deduces _failure oracles_ from existing test suites with labeled outcomes.
_BASHIRI_ is build on execution-feature-driven debugging, that collects features describing the execution of a program to infer a diagnosis. 
Our approach extends this idea combined with causal learning to produce a testing oracle.
   
In our evaluation, the oracles learned by _BASHIRI_ predicted test outcomes with 95% accuracy, demonstrating the approach's effectiveness and quality of the learned oracles.

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