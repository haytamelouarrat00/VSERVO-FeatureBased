# 🎥 Image-Based Visual Servoing (IBVS) Simulator

A complete Python implementation of feature-based visual servoing based on François Chaumette's seminal work. This simulator enables a virtual camera to autonomously navigate to a desired pose by tracking visual features in real-time.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)

![Visual Servoing Demo](demo.gif)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How to Use](#how-to-use)
  - [Basic Examples](#basic-examples)
  - [Using Your Own Images](#using-your-own-images)
  - [Advanced Configuration](#advanced-configuration)
- [Technical Details](#technical-details)
  - [Control Law](#control-law)
  - [Interaction Matrix](#interaction-matrix)
  - [Feature Detection](#feature-detection)
  - [Camera Motion Model](#camera-motion-model)
- [Project Structure](#project-structure)
- [Examples Gallery](#examples-gallery)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

**Visual servoing** is a control technique that uses visual information from cameras to control the motion of a robot or camera system. This project implements **Image-Based Visual Servoing (IBVS)**, where the control is directly performed in the image space using 2D feature coordinates.

### What Does This Do?

Given:
- 🎯 A **desired camera pose** (where you want the camera to be)
- 📷 A **current camera pose** (where the camera is now)
- 🔍 **Visual features** (corners, keypoints in the scene)

The simulator:
1. Extracts and tracks features in both views
2. Computes the error between current and desired feature positions
3. Calculates camera velocities to reduce this error
4. Moves the camera iteratively until convergence

### Why Is This Useful?

- **Robotics**: Autonomous camera positioning, visual tracking
- **Computer Vision**: Understanding camera motion and feature dynamics
- **Education**: Learning control theory and visual servoing principles
- **Research**: Testing new control algorithms and feature detectors

---

## ✨ Features

### Core Capabilities
- ✅ **Multiple Control Laws**: Classic IBVS, adaptive gain, second-order control
- ✅ **Feature Detection Methods**: Harris corners, SIFT keypoints
- ✅ **User Images**: Use your own images as reference targets
- ✅ **Real-time Visualization**: 3D camera motion, error evolution, velocity plots
- ✅ **Scene Generators**: Planar, cube, sphere, grid, custom patterns
- ✅ **Robust Tracking**: Feature correspondence maintenance throughout motion
- ✅ **Safety Limits**: Velocity saturation, visibility checking, singularity handling

### Visualization
- 📊 Real-time error and velocity plots
- 🎬 3D camera trajectory animation
- 👁️ Side-by-side current vs desired views
- 📈 Feature tracking visualization
- 🎯 Convergence monitoring

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
```bash
# Install required packages
pip install numpy scipy matplotlib opencv-python opencv-contrib-python
```

**Detailed requirements:**
```
numpy>=1.19.0
scipy>=1.5.0
matplotlib>=3.3.0
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0  # For SIFT
```

### Clone Repository
```bash
git clone https://github.com/yourusername/visual-servoing-ibvs.git
cd visual-servoing-ibvs
```

### Verify Installation
```bash
python quick_start.py
```

If successful, you'll see a simulation window with a planar scene and real-time plots.

---

## 🎮 Quick Start

### 1. Run Your First Simulation
```bash
python quick_start.py
```

This runs a basic visual servoing simulation with:
- Planar scene (4 corner points)
- Small camera displacement
- Live visualization

### 2. Interactive Menu
```bash
python main.py
```

Select from 14+ pre-configured examples:
```
1. Basic simulation
2. Large displacement
3. Scene comparison
4. Gain comparison
5. Live visualization
...
11. 🆕 Your own image (interactive)
12. 🆕 Your own image (batch)
13. 🆕 Compare multiple images
```

### 3. Use Your Own Image
```bash
python main.py
# Select option 11
# Follow the interactive prompts
```

---

## 📖 How to Use

### Basic Examples

#### Example 1: Standard Simulation
```python
from simulator import SimulatorFactory
from visualizer import LiveVisualizer

# Create simulator
sim = SimulatorFactory.create_standard_simulator(
    scene_type='planar',
    gain=0.5,
    control_law='classic'
)

# Run with visualization
live_vis = LiveVisualizer(sim)
results = live_vis.run_with_visualization(verbose=True)

# Check results
print(f"Converged: {results['converged']}")
print(f"Iterations: {results['iterations']}")
print(f"Final error: {results['final_error']:.6f}")
```

#### Example 2: Custom Scene
```python
from camera import Camera
from virtual_scene import VirtualScene
from simulator import VisualServoingSimulator

# Create custom 3D points
points_3d = np.array([
    [0.5, 0.5, 0],
    [-0.5, 0.5, 0],
    [-0.5, -0.5, 0],
    [0.5, -0.5, 0]
])
scene = VirtualScene(points_3d=points_3d)

# Define camera poses
initial_camera = Camera(position=[0.3, 0.2, -1.5])
desired_camera = Camera(position=[0, 0, -2.0])

# Create simulator
sim = VisualServoingSimulator(scene, initial_camera, desired_camera)
results = sim.run(verbose=True)
```

### Using Your Own Images

#### Interactive Mode (Recommended for First Time)
```python
from user_image_vs import run_user_image_vs_interactive

# Runs interactive setup with prompts
results = run_user_image_vs_interactive()
```

**The interactive mode will:**
1. 📁 Ask for your image path
2. 👁️ Preview detected features
3. ⚙️ Let you configure parameters
4. 🚀 Run the simulation
5. 📊 Show and save results

#### Batch Mode (Quick)
```python
from user_image_vs import run_user_image_vs_batch

results = run_user_image_vs_batch(
    image_path='path/to/your/image.jpg',
    max_features=20,
    gain=0.5,
    displacement='medium',
    show_viz=True
)
```

#### Advanced: Direct SIFT Usage
```python
from sift_features import SIFTImageScene, create_sift_simulator

# Load your image
import cv2
image = cv2.imread('your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create simulator
sim = create_sift_simulator(
    image_array=image,
    max_features=25,
    gain=0.6,
    displacement='medium'
)

# Run
results = sim.run(verbose=True)
```

### Advanced Configuration

#### Custom Control Parameters
```python
controller_params = {
    'gain': 0.7,                    # Control gain λ
    'control_law': 'adaptive',      # 'classic', 'adaptive', 'second_order'
    'depth_estimation': 'desired',  # Depth estimation strategy
    'velocity_limits': {
        'linear': 0.5,              # m/s
        'angular': 0.5              # rad/s
    }
}

simulation_params = {
    'dt': 0.01,                     # Time step (seconds)
    'max_iterations': 1500,
    'convergence_threshold': 1e-3,
    'check_visibility': True,
    'stop_if_features_lost': True
}

sim = VisualServoingSimulator(
    scene, initial_cam, desired_cam,
    controller_params, simulation_params
)
```

#### Custom Feature Detection
```python
from sift_features import SIFTFeatureTracker

# Configure SIFT
tracker = SIFTFeatureTracker(
    n_features=30,
    contrast_threshold=0.03,  # Lower = more features
    edge_threshold=10
)

# Extract features
coords = tracker.extract_reference_features(image)

# Track in new view
current, reference, valid = tracker.track_features(current_image)
```

---

## 🔬 Technical Details

### Control Law

The core of visual servoing is the **exponential decrease of error**:
```
ė = -λ e
```

Where `e = s - s*` is the feature error.

#### Image-Based Visual Servoing (IBVS)

The camera velocity is computed as:
```
v_c = -λ L_s^+ e
```

**Components:**
- `v_c`: Camera velocity (6-DOF: 3 linear + 3 angular)
- `λ`: Control gain (typically 0.1-1.0)
- `L_s^+`: Pseudo-inverse of the interaction matrix
- `e`: Feature error vector (2N dimensions for N points)

**Why does this work?**

The interaction matrix `L_s` relates feature velocities to camera velocities:
```
ṡ = L_s · v_c
```

By controlling `v_c` to make `ṡ = -λe`, we achieve exponential error decrease.

### Interaction Matrix

For a 2D point feature `(x, y)` at depth `Z`, the interaction matrix is:
```
L_s = [ -1/Z    0      x/Z    xy       -(1+x²)   y   ]
      [  0     -1/Z    y/Z    1+y²     -xy       -x  ]
```

**Interpretation:**
- **Columns 1-3**: Effect of linear velocities (translation)
- **Columns 4-6**: Effect of angular velocities (rotation)

**For N points**, we stack N interaction matrices:
```python
L = np.vstack([L_point1, L_point2, ..., L_pointN])  # Shape: (2N, 6)
```

#### Example: Single Point

Consider a point at `(x, y, Z) = (0.1, -0.2, 2.0)`:
```python
L = [[-0.5,    0,     0.05,   -0.02,   -1.01,    -0.2  ]
     [ 0,     -0.5,   -0.1,    0.96,    0.02,    -0.1  ]]
```

If we apply a forward velocity `v_z = 0.1 m/s`:
```
ṡ = L · [0, 0, 0.1, 0, 0, 0]^T
  = [0.005, -0.01]^T
```
The point moves right and down in the image (perspective effect).

### Feature Detection

#### Harris Corner Detection

**Algorithm:**
1. Compute image gradients `I_x`, `I_y`
2. Build structure tensor:
```
   M = [ Σ(I_x²)   Σ(I_x·I_y) ]
       [ Σ(I_x·I_y) Σ(I_y²)   ]
```
3. Compute corner response:
```
   R = det(M) - k·trace(M)²
```
4. Threshold and non-maximum suppression

**Advantages:**
- Fast computation
- Rotation invariant
- Good for structured scenes

**Limitations:**
- No descriptor (can't track across large motions)
- Scale dependent

#### SIFT Feature Detection

**Algorithm:**
1. Build scale-space pyramid
2. Detect DoG (Difference of Gaussians) extrema
3. Localize keypoints with sub-pixel accuracy
4. Assign orientation
5. Compute 128-D descriptor

**Advantages:**
- Scale invariant
- Rotation invariant
- Robust descriptors for matching
- Can track features across large motions

**Key Concept:** **Fixed Feature Correspondence**
```python
# ✅ CORRECT: Extract once, track throughout
reference_features = sift.detect(desired_image)  # Done ONCE

for iteration in control_loop:
    current_features = match_and_track(current_image, reference_features)
    error = current_features - reference_features  # Same features!
    velocity = compute_control(error)
```
```python
# ❌ WRONG: Re-detect every time
for iteration in control_loop:
    current_features = sift.detect(current_image)  # Different features!
    desired_features = sift.detect(desired_image)
    error = current_features - desired_features  # Meaningless!
```

**Why?** IBVS requires that `s*` (desired features) remains **fixed** and we track the **same features** in the current view.

### Camera Motion Model

#### Coordinate Frames

- **World Frame**: Fixed reference frame
- **Camera Frame**: Attached to camera (Z-axis = optical axis)

#### Camera Pose

A camera pose is defined by:
```
T = [R | t]
    [0 | 1]
```

Where:
- `R`: 3×3 rotation matrix (orientation)
- `t`: 3×1 translation vector (position)

#### Velocity Integration

Camera velocity `v_c = [v_x, v_y, v_z, ω_x, ω_y, ω_z]^T` is integrated using:

**Translation Update:**
```python
# Velocity is in camera frame, convert to world frame
v_world = R @ v_camera
position_new = position_old + v_world * dt
```

**Rotation Update (Exponential Map):**
```python
# ω = [ω_x, ω_y, ω_z] is angular velocity
θ = ||ω||  # Rotation angle
if θ > 0:
    k = ω / θ  # Rotation axis (unit vector)
    
    # Rodrigues' formula
    K = skew_symmetric(k)
    ΔR = I + sin(θ·dt)·K + (1-cos(θ·dt))·K²
    
    # Update rotation
    R_new = R_old @ ΔR
```

**Why exponential map?**
- Preserves rotation matrix properties (orthogonal, det=1)
- Avoids gimbal lock
- Smooth interpolation

#### Projection Model

**3D Point to Image:**
```
[u]   [f  0  c_x] [X/Z]
[v] = [0  f  c_y] [Y/Z]
[1]   [0  0   1 ] [ 1 ]
```

Where:
- `(X, Y, Z)`: 3D point in camera frame
- `(u, v)`: 2D pixel coordinates
- `f`: Focal length
- `(c_x, c_y)`: Principal point (image center)

**Normalized Coordinates:**
```
x = X/Z = (u - c_x) / f
y = Y/Z = (v - c_y) / f
```

Used in the interaction matrix computation.

### Depth Estimation Strategies

Since we often don't know the exact current depth `Z`, we use:

1. **Desired Depth** (most common):
```python
   Z_current ≈ Z_desired
```
   Assumes depth doesn't change much during motion.

2. **Mean Depth**:
```python
   Z_current = mean(Z_desired_all_points)
```

3. **Adaptive**:
```python
   Z_current = Z_desired * (||s_desired|| / ||s_current||)
```
   Scales based on feature displacement.

### Convergence Criteria

Simulation stops when:

1. **Error threshold**:
```
   ||e|| < ε  (typically 1e-3)
```

2. **Sustained convergence**:
```
   ||e|| < ε  for last N iterations (N=5)
```

3. **Maximum iterations reached**

4. **Features lost** (visibility check)

---

## 📁 Project Structure
```
visual-servoing-ibvs/
│
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
│
├── 🐍 Core Modules
│   ├── camera.py                   # Camera model, projections, pose updates
│   ├── virtual_scene.py            # 3D scene management
│   ├── interaction_matrix.py      # Image Jacobian computation
│   ├── vs_controller.py            # Control law implementations
│   ├── simulator.py                # Main simulation orchestrator
│   └── visualizer.py              # Real-time plotting and 3D viz
│
├── 🎨 Feature Detection
│   ├── image_features.py           # Harris corner detection
│   ├── sift_features.py            # SIFT detection and tracking
│   └── user_image_vs.py           # User image handling
│
├── 🎯 Examples & Demos
│   ├── main.py                     # Main menu with all examples
│   ├── quick_start.py              # Quick test script
│   ├── demo_image_features.py      # Image feature demos
│   └── image_based_simulator.py    # Image-based simulation wrapper
│
├── 📊 Output (generated)
│   ├── *_plots.png                # Error and velocity plots
│   ├── *_3d.png                   # 3D visualization
│   └── *_features.png             # Detected features
│
└── 📖 Documentation
    ├── docs/
    │   ├── theory.md              # Detailed theory
    │   ├── examples.md            # Usage examples
    │   └── api.md                 # API reference
    └── images/                     # Demo images and screenshots
```

### Key Components

#### `camera.py`
- Camera intrinsic/extrinsic parameters
- 3D ↔ 2D projections
- Pose transformations
- Velocity-based updates

#### `interaction_matrix.py`
- Computes L_s for point features
- Pseudo-inverse computation
- Depth estimation strategies
- Singularity detection

#### `vs_controller.py`
- Classic IBVS control law
- Adaptive gain control
- Velocity limiting
- Convergence detection

#### `simulator.py`
- Main control loop
- Feature visibility checking
- History tracking
- Results compilation

#### `visualizer.py`
- Real-time error/velocity plots
- 3D scene visualization
- Camera trajectory animation
- Feature correspondence display

---

## 🖼️ Examples Gallery

### Example 1: Basic Planar Scene
![Basic Example](images/example1_combined.png)
- 4 corner points
- Small displacement
- Classic control

### Example 2: Large Displacement
![Large Displacement](images/example2_combined.png)
- Significant initial error
- Adaptive gain control
- Smooth convergence

### Example 3: SIFT Features
![SIFT Features](images/example_sift.png)
- Harris corners vs SIFT
- Feature tracking visualization
- Robust matching

### Example 4: User Images
![User Image](images/example_user_image.png)
- Custom textured image
- 20+ SIFT keypoints
- Real-world scenario

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "No SIFT features detected"

**Cause:** Image lacks texture or contrast

**Solutions:**
```python
# Try lowering the contrast threshold
tracker = SIFTFeatureTracker(
    contrast_threshold=0.02,  # Lower = more features
    edge_threshold=15
)

# Or use a more textured image
```

#### 2. "Simulation not converging"

**Cause:** Gain too high, poor feature distribution, or large displacement

**Solutions:**
```python
# Lower the gain
controller_params = {'gain': 0.3}  # Instead of 0.5

# Start with smaller displacement
displacement = 'small'  # Instead of 'large'

# Increase features
max_features = 25  # Instead of 15
```

#### 3. "Features lost during motion"

**Cause:** Features go out of camera field of view

**Solutions:**
```python
# Use smaller initial displacement
# Increase velocity limits carefully
# Check feature distribution (should be spread out)

# Enable visibility checking
simulation_params = {
    'check_visibility': True,
    'stop_if_features_lost': True
}
```

#### 4. "ImportError: No module named cv2"

**Solution:**
```bash
pip install opencv-python opencv-contrib-python
```

#### 5. "Slow visualization"

**Solutions:**
```python
# Reduce update frequency
# In visualizer.py, change:
update_frequency = 20  # Instead of 10

# Or run without live visualization
results = sim.run(verbose=True)
# Then visualize after
```

### Performance Tips

**For faster simulations:**
- Reduce `dt` (time step)
- Use fewer features (15-20 is usually sufficient)
- Disable live visualization
- Use Harris instead of SIFT (faster but less robust)

**For better convergence:**
- Start with small displacement
- Use medium gain (0.4-0.6)
- Ensure features are well distributed
- Use 15-25 features
- Enable visibility checking

---

## 📚 References

### Seminal Papers

1. **Chaumette, F., & Hutchinson, S. (2006).**  
   "Visual servo control, Part I: Basic approaches"  
   *IEEE Robotics & Automation Magazine*, 13(4), 82-90.
   - Foundation of IBVS and PBVS

2. **Chaumette, F., & Hutchinson, S. (2007).**  
   "Visual servo control, Part II: Advanced approaches"  
   *IEEE Robotics & Automation Magazine*, 14(1), 109-118.
   - Advanced techniques and hybrid approaches

3. **Espiau, B., Chaumette, F., & Rives, P. (1992).**  
   "A new approach to visual servoing in robotics"  
   *IEEE Transactions on Robotics and Automation*, 8(3), 313-326.
   - Task function approach

4. **Lowe, D. G. (2004).**  
   "Distinctive image features from scale-invariant keypoints"  
   *International Journal of Computer Vision*, 60(2), 91-110.
   - SIFT feature detection

### Books

- **Corke, P. (2017).** *Robotics, Vision and Control* (2nd ed.). Springer.
- **Hutchinson, S., Hager, G. D., & Corke, P. I. (1996).** "A tutorial on visual servo control." *IEEE transactions on robotics and automation*, 12(5), 651-670.

### Online Resources

- [ViSP (Visual Servoing Platform)](https://visp.inria.fr/)
- [Peter Corke's Robotics Toolbox](https://github.com/petercorke/robotics-toolbox-python)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Bug Reports**: Open an issue with details
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve or translate documentation
5. **Examples**: Share interesting use cases

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/visual-servoing-ibvs.git
cd visual-servoing-ibvs

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit (`git commit -m 'Add amazing feature'`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to functions and classes
- Comment complex algorithms

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **François Chaumette** for pioneering work in visual servoing
- **Peter Corke** for excellent robotics educational resources
- **OpenCV community** for robust computer vision tools
- All contributors and users of this project

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/visual-servoing-ibvs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/visual-servoing-ibvs/discussions)
- **Email**: your.email@example.com

---


## 📈 Roadmap

### Version 1.0 (Current)
- ✅ Basic IBVS implementation
- ✅ Harris corner detection
- ✅ SIFT feature tracking
- ✅ User image support
- ✅ Real-time visualization

### Version 1.1 (Planned)
- [ ] ORB feature detection
- [ ] KLT feature tracking
- [ ] Position-Based VS (PBVS)
- [ ] Hybrid visual servoing
- [ ] ROS integration

### Version 2.0 (Future)
- [ ] Real camera support (USB/IP cameras)
- [ ] 3D object models
- [ ] Multi-camera systems
- [ ] Deep learning features
- [ ] Real robot integration

---

<div align="center">

**Made with ❤️ for robotics and computer vision education**

[⬆ Back to Top](#-image-based-visual-servoing-ibvs-simulator)

</div>