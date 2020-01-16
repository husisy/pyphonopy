# pyphonopy

A pure python implementation package to calculate phonon spectrum from vasprun force constant.

The package phonopy (see link) is quite looooong and obscure with a lot unnecessary c/c++ code.

## quickstart

1. download the whole repository: git clone or download as zip
2. at the root folder, you should see `data setup.py pyphonopy test_pyphonopy.py utils.py README.md`
3. run unittest at the root folder: `pytest`
   * besides standard libraries (STL), `numpy` is the only requirement to run pyphonopy
   * but in order to pass the unittest, you need install `pytest periodictable pyyaml lxml` via `pip` or `conda`
4. install the `pyphonopy` into the global package list
   * run commands at the root folder: `pip install .`

## misc

Why not provide command tools just like phonopy?

> Writing scripts to do these tasks is my preferred way, which enables third-party tools development.

Will the pure python implementation be much slower than the c/c++ optimization code in phonopy?

> Personally I believe the numpy performace.
