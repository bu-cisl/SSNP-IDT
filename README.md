# SSNP-IDT: Split-step non-paraxial model for intensity diffraction tomography

Python implementation of paper: **High-fidelity intensity diffraction tomography with a non-paraxial multiple-scattering model**.
This repository includes a highly flexible and easy-to-use Python package based on PyCUDA library, and code examples for simulation and reconstruction.

## Overview



## Installation

1. Prepare the environment following the [pre-installation steps for PyCUDA](https://wiki.tiker.net/PyCuda/Installation/) on a computer with NVIDIA GPU
2. Install the `ssnp` package in this repository
```shell
pip install "git+https://github.com/bu-cisl/SSNP-IDT#subdirectory=ssnp_pkg"
```
3. Download the examples and run with Python

## License

Project is licensed under the terms of the GPL-v3 license. see the LICENSE file for details

## Reference

Sharma, A., & Agrawal, A. (2004). Split-step non-paraxial beam propagation method. *Physics and Simulation of Optoelectronic Devices XII*, 5349, 132. https://doi.org/10.1117/12.528172

Lim, J., Ayoub, A. B., Antoine, E. E., & Psaltis, D. (2019). High-fidelity optical diffraction tomography of multiple scattering samples. *Light: Science & Applications*, 8(1), 82. https://doi.org/10.1038/s41377-019-0195-1
