
# MPLC

A python code simulating the Multi-Plane Light Conversion (MPLC) technology based on the article https://doi.org/10.1038/s41467-019-09840-4

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MPLC

```bash
pip install MPLC
```

## Usage

The package comes with a jupyter notebook called "example.ipynb", to exemplify the usage of the cpde.

```python
from MPLC.system import System

# Start the system with the default variables
s = System()

# Create Gaussian beams with 5 modes ( 15 beams ) in a triangle position
Gaussian = s.create_Gaussians(num_modes=5, positions="triangle")

# Plot the Gaussians
fig,ax = Gaussian.plot()

# Create Hermite Gaussian beams

Hermite = s.create_HermiteGaussian()

# Set the beams in the system
s.set_fields(Gaussian,Hermite)

# Start the process of compressing the Gaussians into Hermite Gaussian beams
s.start()

# Calculate the matrix with single values of the transfer matrix, the insertion loss and the mode-dependent loss
s, IL, MDL = s.get_couplingMatrix()
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.


## License

[CC 1.0 universal](https://creativecommons.org/publicdomain/zero/1.0/deed.pt)

## Author

Joao Abreu, joaocarloscabreu@gmail.com