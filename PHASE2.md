# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## 1. Containerization
- [x] **1.1 Dockerfile**
  - [x] Dockerfile created and tested
  - [x] Instructions for building and running the container
- [x] **1.2 Environment Consistency**
  - [x] All dependencies included in the container

## 2. Monitoring & Debugging

- [x] **2.1 Debugging Practices**
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
  
## 3. Profiling & Optimization
- [x] **3.1 Profiling Scripts**
  - [x] cProfile, PyTorch Profiler, or similar used
  - [x] Profiling results and optimizations documented
  
### Tools Used
- **cProfile**: Used for lightweight CPU-based profiling to inspect function call times.
- **SnakeViz**: Web-based visualization of `.prof` files to help identify runtime bottlenecks.
- **PyTorch Profiler**: Enabled layer-wise and GPU/CPU operation profiling for model components.
- **TensorBoard**: Visualized PyTorch profiling traces in an intuitive timeline format.

### Commands Used
```bash
# General CPU profiling
python -m cProfile -s cumtime image_caption/main.py
python -m cProfile -o profile_results.prof image_caption/main.py

# Visualization
pip install snakeviz
snakeviz profile_results.prof

# Install PyTorch and TensorBoard
pip install torch torchvision tensorboard

# Run PyTorch Profiler script
python image_caption/pytorch_profiler.py
```

## 4. Experiment Management & Tracking
- [x] **4.1 Experiment Tracking Tools**
  - [x] MLflow, Weights & Biases, or similar integrated
  - [x] Logging of metrics, parameters, and models
  - [x] Instructions for visualizing and comparing runs

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
- [x] **7.1 Updated README**
  - [x] Instructions for all new tools and processes
  - [x] All scripts and configs included in repo
