DAW RF: Data AWare Random Forests
---
<!--[![PyPi version](https://img.shields.io/pypi/v/dare-rf)](https://pypi.org/project/dare-rf/)-->
<!--[![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/dare-rf/)-->
[![Github License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/jjbrophy47/dare_rf/blob/master/LICENSE)
<!--[![Build](https://github.com/jjbrophy47/dare_rf/actions/workflows/wheels.yml/badge.svg?branch=v1.0.0)](https://github.com/jjbrophy47/dare_rf/actions/workflows/wheels.yml)-->

**daw-rf** is a python library that builds random forest models and keeps statistics about the data in the model to enable post-hoc adversarial robustness analysis.

<!--<p align="center">
	<img align="center" src="images/thumbnail.png" alt="thumbnail", width="350">
</p>-->

<!--Installation
---
```sh
pip install dare-rf
```-->

<!--Usage
---
Simple example of removing a single training instance:

```python
import dare
import numpy as np

# training data
X_train = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]])
y_train = np.array([1, 1, 1, 0, 1])

X_test = np.array([[1, 0]])  # test instance

# train a DaRE RF model
rf = dare.Forest(n_estimators=100,
                 max_depth=3,
                 k=5,  # no. thresholds to consider per attribute
                 topd=0,  # no. random node layers
                 random_state=1)
rf.fit(X_train, y_train)

rf.predict_proba(X_test)  # prediction before deletion => [0.5, 0.5]
rf.delete(3)  # delete training example at index 3 ([1, 0], 0)
rf.predict_proba(X_test)  # prediction after deletion => [0.0, 1.0]
```-->
