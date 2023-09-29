# bashiri

Bashir is an approach to infer failure oracles from a small test suite automatically.

## Abstract

When fixing a program - be it manually or through automated repair - it is crucial that the fix (a) entirely fixes previously failing runs and (b) does not impact previously passing runs.
Both properties are typically validated by a *test suite*.
However, such validation still brings the risks of *overspecialization* (the patch overfitting the failing test(s)) and *overgeneralization* (the patch affecting passing runs not in the test suite).
These threats are especially present for *functional* failures, less noticeable than crashes or hangs.

We introduce *bashiri*, an approach that deduces *failure oracles* from existing test suites with labeled outcomes.
bashiri gathers features during initial test case executions (like coverage of specific lines or variable relationships).
From these features, bashiri trains a *model* as an oracle capable of predicting test outcomes for unseen, non-crashing runs:
"The failure occurs if Line~6 is executed and y < x holds".

In our evaluation, the oracles learned by bashiri predicted test outcomes with 87% accuracy.
For two-thirds of real-world subjects investigated, the bashiri oracles achieved 100% correctness when differentiating passing and failing test scenarios due to bashiri's fine-grained features.
The resulting oracles can be easily read and assessed by humans, who can use them to assess the quality of fixes, automatically predict software failures, guide test generation towards likely failures, and more.

## Usage

For bashiri, you need to instrument your subject.
```python
instrument("middle.py", "tmp.py")
```
Next, you need some tests to execute and collect their event traces. 
We provide two collectors, one for unit tests and one for input to the program.
However, implementing another collector by inheriting the base class `EventCollector` and implementing its `collect()` method is an option.
To employ the collector, use it like this:
```python
collector = SystemtestEventCollector(os.getcwd(), "tmp.py")
events = collector.get_events((passing, failing))
```
In this example, we leverage the input event collector. `passing` and `failing` are lists of passing and failing inputs, respectively.

Next, you can utilize the event handler to extract and build feature vectors from the event traces.
```python
handler = EventHandler()
handler.handle_files(events)
```

Now, we can leverage bashiri's learning to infer a failure oracle.
```python
oracle = DecisionTreeOracle()
oracle.fit(
    handler.feature_builder.get_all_features(),
    handler.feature_builder.get_vectors(),
)
``` 

Now, we can leverage the collector, handle, and oracle to identify the result of an unseen execution/test case with high accuracy.

We provide an example of this walk-trough in `evaluation/example.ipynb`.


## Explainibility

We have incorporated mechanisms to explain oracles by analyzing their underlying model or interpreting them using SHAP[1].
```python
from sklearn import tree

explanation = oracle.explain()
tree.plot_tree(oracle.model)
```

## Human-In-The-Loop

We have additionally implemented a mechanism to incorporate a human oracle to enrich insufficient test sets in the `HumanOracleRefinement` class.
```python
seeds = {i: s, for i, s in enumerate(passing + failing)}
refinement = HumanOracleRefinement(handler, oracle, seeds, collector)
refinement.run()
```