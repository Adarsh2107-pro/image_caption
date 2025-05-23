# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## 1. Containerization
- [ ] **1.1 Dockerfile**
  - [x] Dockerfile created and tested
  - [x] Instructions for building and running the container
- [x] **1.2 Environment Consistency**
  - [x] All dependencies included in the container

## 2. Monitoring & Debugging

- [ ] **2.1 Debugging Practices**
  - [x] Debugging tools used (e.g., pdb)
  - [x] Example debugging scenarios and solutions

#### Tools Used
- **pdb**: Used in early development to pause execution and inspect variables.
- **print() statements**: Employed during preprocessing and model training to validate input/output shapes.
- **TensorFlow logs**: Utilized to monitor warnings and catch shape mismatches.

#### Issues Faced
No critical runtime errors were encountered during training or inference.

#### Preparedness
Debugging breakpoints (`pdb.set_trace()`) and manual print statements were introduced in the early development stages to inspect data flow and model behavior. These were removed or commented out after achieving stability.

#### Example
```python
# Used for tensor shape inspection
print(f"X1: {X1.shape}, X2: {X2.shape}, y: {y.shape}")

## 3. Profiling & Optimization
- [ ] **3.1 Profiling Scripts**
  - [ ] cProfile, PyTorch Profiler, or similar used
  - [ ] Profiling results and optimizations documented

## 4. Experiment Management & Tracking
- [ ] **4.1 Experiment Tracking Tools**
  - [ ] MLflow, Weights & Biases, or similar integrated
  - [ ] Logging of metrics, parameters, and models
  - [ ] Instructions for visualizing and comparing runs

## 5. Application & Experiment Logging
- [x] **5.1 Logging Setup**
  - [x] logger and/or rich integrated

`logger` with rich can be imported from `image_caption.config`. A logs/ directory will be created in the parent directory of the script where it is used, but you may change it in image_caption/config.py. `rich` is used to enhance logs. Use logger.info() or logger.error() to write to a running output log in these directories. See docs/info.md for more detail.

  - [x] Example log entries and their meaning

## 6. Configuration Management
- [x] **6.1 Hydra or Similar**
  - [x] Configuration files created
  - [x] Example of running experiments with different configs

## 7. Documentation & Repository Updates
- [ ] **7.1 Updated README**
  - [ ] Instructions for all new tools and processes
  - [ ] All scripts and configs included in repo
