# Introduction

## evclust : An Python library for evidential clustering

[![Documentation Status](https://readthedocs.org/projects/evclust/badge/?version=latest)](https://evclust.readthedocs.io/en/latest/)
[![tests](https://github.com/py-pkgs/py-pkgs-cookiecutter/workflows/test/badge.svg)](https://github.com/py-pkgs/py-pkgs-cookiecutter/workflows/test/badge.svg)
[![release](https://img.shields.io/github/v/release/armelsoubeiga/evclust.svg)](https://github.com/armelsoubeiga/evclust/releases)
[![python](https://img.shields.io/badge/python-%5E3.8-blue)](https://img.shields.io/badge/python-%5E3.8-blue)
[![os](https://img.shields.io/badge/OS-Ubuntu%2C%20Mac%2C%20Windows-purple)](https://img.shields.io/badge/OS-Ubuntu%2C%20Mac%2C%20Windows-purple)

<br/>

<div align="center">

<img src="https://raw.githubusercontent.com/armelsoubeiga/evclust/master/docs/assets/logo.png" height="180px" />

[Get Started](https://evclust.readthedocs.io) | [pip install](https://pypi.org/project/evclust/)

</div>

<br/>

Various clustering algorithms that produce a credal partition, i.e., a set of Dempster-Shafer mass functions representing the membership of objects to clusters. The mass functions quantify the cluster-membership uncertainty of the objects. 

## News
* More than 10 uncertainty-based clustering algorithms. [See full list](https://evclust.readthedocs.io/en/latest/autoapi/available/index.html). 

* Coming soon :
    * More notebook examples 
    * Belief Shift Clustering (BSC), 
    * Dynamic evidential c-means clustering (DECM), 
    * Deep Evidential Clustering (DEC),
    * Decision tree-based evidential clustering (DTEC),
    * Transfer learning-based evidential c-means clustering (TECM),
    * Etc ..

## Informations

* Title: Evidential Clustering
* Version: 0.2 -- Date: 2024-29-11
* Previous version: 
    - 0.1.5 -- Date: 2024-07-17
    - 0.1.1 -- Date: 2023-09-01
* License: MIT
* Depends: Pyhton >=3.8
* Author: [Armel SOUBEIGA](https://armelsoubeiga.github.io/)
* Maintainer: [armel.soubeiga@uca.fr](https://armelsoubeiga.github.io/)
* Contributors : [Violaine ANTOINE](https://perso.isima.fr/~viantoin/)


## Citation 

```
Paper in preparation ...

@article{soubeiga2024,
  title={evclust : An Python library for evidential clustering},
  author={Soubeiga, Armel and Antoine, Violaine},
  journal={arXiv preprint arXiv:~~},
  year={2024}
}
```

## References

* [1] Denœux, T. (Year). evclust: An R Package for Evidential Clustering. Université de technologie de Compiègne. url [https://cran.r-project.org/web/packages/evclust/index.html](https://cran.r-project.org/web/packages/evclust/index.html)


## Installation

```bash
$ pip install evclust 
```

## Usage

For example, `ecm` computes a credal partition from a matrix of attribute data using the Evidential c-means (ECM) algorithm.

```python
# Import test data
from evclust.datasets import load_decathlon, load_iris
df = load_iris()
df.head()
df=df.drop(['species'], axis = 1)

# Evidential clustering with c=3
from evclust.ecm import ecm
model = ecm(x=df, c=3,beta = 1.1,  alpha=0.1, delta=9)

# Read the output
from evclust.utils import ev_summary, ev_pcaplot
ev_summary(model)    
ev_pcaplot(data=df, x=model, normalize=False)    
```

## Descriptions

Evidential clustering is a modern approach in clustering algorithms that addresses uncertainty in group membership by employing the Dempster-Shafer theory. This approach yields a credal partition, represented by a tuple of mass functions, which captures the uncertain assignment of objects to clusters.

This package offers efficient algorithms for evidential clustering. The package provides functions for visualizing, evaluating, and utilizing credal partitions, allowing for a comprehensive analysis of uncertain group assignments. 

**evclust** is referenced by [**The Belief Functions and Applications Society (BFAS)**](https://bfasociety.org/)

**evclust** is also available in **R** by [**Thierry Denoeux**](https://cran.rstudio.com/web/packages/evclust/vignettes/evclust_vignette.pdf)


## Contributing

Interested in contributing? Check out the [Contributing Guidelines](https://evclust.readthedocs.io/en/latest/contributing.html). Please note that this project is released with a [Code of Conduct](https://evclust.readthedocs.io/en/latest/conduct.html). By contributing to this project, you agree to abide by its terms.

## License

`evclust` was created by Armel SOUBEIGA. It is licensed under the terms of the MIT license.
