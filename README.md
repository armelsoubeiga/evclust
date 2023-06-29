# evclust: Evidential c-Means Clustering

[![Documentation Status](https://readthedocs.org/projects/py-pkgs-cookiecutter/badge/?version=latest)](https://py-pkgs-cookiecutter.readthedocs.io/en/latest/)
![tests](https://github.com/py-pkgs/py-pkgs-cookiecutter/workflows/test/badge.svg)
[![release](https://img.shields.io/github/release/py-pkgs/py-pkgs-cookiecutter.svg)](https://github.com/py-pkgs/py-pkgs-cookiecutter/releases)
[![python](https://img.shields.io/badge/python-%5E3.8-blue)]()
[![os](https://img.shields.io/badge/OS-Ubuntu%2C%20Mac%2C%20Windows-purple)]()

<br/>

Various clustering algorithms that produce a credal partition, i.e., a set of Dempster-Shafer mass functions representing the membership of objects to clusters. The mass functions quantify the cluster-membership uncertainty of the objects. The algorithms are: Evidential c-Means, Relational Evidential c-Means, Constrained Evidential c-Means, Multiples Relational Evidential c-Means. 

Example: consider a set of n = 5 individuals and c = 3 clusters, we have the following credal partition :
<style>
    .table-wrapper {
        display: inline-block;
        vertical-align: top;
    }
    .image-wrapper {
        display: inline-block;
        vertical-align: top;
    }
</style>

<div class="table-wrapper">

|         | m\_1(A) | m\_2(A) | m\_3(A) | m\_4(A) |
|---------|-------|-------|-------|-------|
| ∅       | 1     | 0     | 0     | 0     |
| {ω1}    | 0     | 0     | 0.2   | 0     |
| {ω2}    | 0     | 1     | 0.4   | 0     |
| {ω1,ω2} | 0     | 0     | 0     | 0     |
| {ω3}    | 0     | 0     | 0.4   | 0     |
| {ω1,ω3} | 0     | 0     | 0     | 0     |
| {ω2,ω3} | 0     | 0     | 0     | 0     |
| Ω       | 0     | 0     | 0     | 1     |

</div>

<div class="image-wrapper">

![Image 1](docs/imgs/image1.jpg)
![Image 2](docs/imgs/image2.jpg)
![Image 3](docs/imgs/image3.jpg)

</div>

## Informations

* Title: Evidential c-Means Clustering
* Version: 0.1
* Date: 2023-06-26
* License: GPL-3
* Depends: Pyhton (>= 3)
* Author: [Armel SOUBEIGA](https://armelsoubeiga.github.io/)
* Maintainer: Armel SOUBEIGA, [armel.soubeiga@uca.fr](armel.soubeiga@uca.fr)
* Contributors : [Violaine ANTOINE](https://perso.isima.fr/~viantoin/) 


## References

* [1] Denœux, T. (Year). evclust: An R Package for Evidential Clustering. Université de technologie de Compiègne. url [https://cran.r-project.org/web/packages/evclust/index.html](https://cran.r-project.org/web/packages/evclust/index.html)
* [2] Masson, M. H. et T. Denœux (2008). ECM : An evidential version of the fuzzy c-means algorithm. Pattern Recognition 41, 1384–1397


## Installation

```bash
$ pip install evclust
```

## Contributing

Interested in contributing? Check out the [Contributing Guidelines](https://evclust.readthedocs.io/en/latest/contributing.html). Please note that this project is released with a [Code of Conduct](https://evclust.readthedocs.io/en/latest/conduct.html). By contributing to this project, you agree to abide by its terms.

## License

`evclust` was created by Armel SOUBEIGA. It is licensed under the terms of the MIT license.
