# BAT_jax.py

## Overview

`BAT_jax.py` converts molecular Cartesian coordinates into Bond-Angle-Torsion (BAT) coordinates using JAX and MDAnalysis.

The file is designed for molecular dynamics and machine learning workflows where molecular structures are represented using:
- bond lengths,
- bond angles,
- and torsion angles

instead of raw XYZ coordinates.

The module also supports reconstructing Cartesian coordinates back from BAT coordinates.

---

# What the File Does

The system:
- reads molecular atom positions,
- computes internal molecular geometry,
- stores the structure in BAT coordinate form,
- and reconstructs the original 3D structure when needed.

This representation is useful because internal coordinates are often:
- more compact,
- more stable,
- and easier for machine learning models to learn.

---

# Main Features

- Converts Cartesian coordinates → BAT coordinates
- Computes:
  - bond lengths
  - bond angles
  - torsion angles
- Reconstructs Cartesian coordinates from BAT representation
- Uses JAX for numerical computation
- Supports molecular dynamics trajectories

---

# Main Workflow

```text
Cartesian Coordinates
        ↓
BAT Coordinate Conversion
        ↓
Bond / Angle / Torsion Representation
        ↓
Machine Learning or Molecular Processing
        ↓
Cartesian Reconstruction
```

---

# Key Functions

## `_find_torsions()`

Builds torsion-angle relationships between bonded atoms.

---

## `_single_frame()`

Converts a molecular structure from Cartesian coordinates into BAT coordinates.

---

## `Cartesian()`

Reconstructs 3D Cartesian coordinates from BAT coordinates.

---

# Technologies Used

- JAX
- MDAnalysis
- NumPy-style tensor operations
- Molecular coordinate transformations

---

# Applications

This module can be used for:
- molecular structure learning
- protein modeling
- molecular autoencoders
- scientific machine learning
- coordinate transformation systems

# custom_encoder_jax.py

## Overview

`custom_encoder_jax.py` implements a customizable encoder and lightweight autoencoder architecture using JAX.

The system transforms high-dimensional input data into compact latent-space representations for machine learning tasks such as:
- dimensionality reduction,
- representation learning,
- and scientific data compression.

---

# What the File Does

The module:
- builds configurable neural network encoders,
- applies hidden-layer transformations,
- supports activation functions and dropout,
- and generates compressed latent vectors from input data.

---

# Main Components

## `BaseEncoder`

Defines the shared encoder structure.

Includes:
- hidden layers
- activation functions
- dropout support

---

## `CustomEncoder`

Implements the encoding process.

The encoder:
- passes input through hidden layers,
- applies nonlinear activation functions,
- and outputs latent-space representations.

---

## `CustomAutoEncoder`

Wraps the encoder into an autoencoder-style architecture.

The model:
- accepts input data,
- encodes it into latent vectors,
- and returns compressed feature representations.

---

# Model Workflow

```text
Input Data
    ↓
Dense Hidden Layers
    ↓
Activation Functions
    ↓
Optional Dropout
    ↓
Latent Representation
```

---

# Features

- Configurable hidden layer sizes
- Custom latent dimensions
- Activation function support
- Dropout regularization
- Modular encoder design
- JAX-based tensor computation

---

# Technologies Used

- JAX
- Neural network dense layers
- Tensor operations
- Latent-space encoding

---

# Example Usage

```python
model = CustomAutoEncoder(
    input_size=128,
    hidden_layers=[64, 32],
    n_latents=16
)

latent_vector = model(x)
```

---

# Applications

This module can be used for:
- representation learning
- dimensionality reduction
- molecular machine learning
- scientific AI pipelines
- latent-space modeling
