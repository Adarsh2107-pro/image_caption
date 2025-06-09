# Image Caption Generation MLOps Pipeline

## 1. Team Information
- [x] Team Name: Team 2
- [x] Team Members: Aadarsh Kumar Narain (anarian2@depaul.edu), Aytaj Mahammadli (amahamma@depaul.edu), Josh Knize (jknize1@depaul.edu)
- [x] Course & Section: SE 489 - Section 910

## 2. Project Overview
- [x] Brief summary of the project (2-3 sentences)

This project aims to provide a complete machine learning (ML) solution to address the problem of image caption generation. This repository can be used to deploy a custom solution consisting of a convolutional neural network (CNN) and long short-term memory (LSTM) network in an MLOps pipeline.

- [x] Problem statement and motivation

The problem of image caption generation is currently of great interest in artificial intelligence (AI) research. Recent advancements in image classification and object detection, as well as natural language processing (NLP) with large language models (LLMs), has opened up opportunities for further research into automatic image caption generation. The ability to do this task can lead to significant progress in other problems such as image searchability, visual question answering (VQA), image understanding, and multi-modal model capabilities.

- [x] Main objectives

Our main project objectives are:
    - Deploy an image captioning ML model for end users to interact with
    - Develop a continuously integrated repository of organized and well-documented code and data
    - Use proper version control for our codebase, models, and data
    - Ensure reproducibility across our experiments and repository

## 3. Project Architecture Diagram
- [x] Insert or link to your architecture diagram (e.g., draw.io, PNG, etc.)

See architecture diagram in docs/architecture.png

## 4. Phase Deliverables
- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions
- [x] How to set up the environment (conda/pip, requirements.txt, Docker, etc.)

If you want to work within the repository, follow the steps below:

1. Clone the Github repository and open the directory in the command line.

2. Create a conda environment:
    - `conda create -n image_caption_env python=3.10`
3. Activate the conda environment and install dependencies:
    - `conda activate image_caption_env`
    - `pip install -r requirements.txt`

- [x] How to run the code and reproduce results

If you want to replicate our results, then Docker can be used for training and evaluating the model.

1. Install Docker using this link: https://docs.docker.com/get-started/get-docker/

2. Build the Docker image with this command: `docker build -f dockerfiles/main.dockerfile . -t main:latest`

3. Download the data using the DVC instructions. Alternatively, download using Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k

4. Run the Docker image
 ```bash
docker run --name exp1 --rm ^
-v /"Path to repo on your local machine"/flickr8k:/flickr8k ^
-v /"Path to repo on your local machine"/config:/config ^
-v /"Path to repo on your local machine"/models:/models ^
main:latest
```

## 6. Contribution Summary
- [ ] Briefly describe each team member's contributions

Aadarsh Narain:
    - Created Git repository and structure.
    - Uploaded models, data, and code.
    - Created DVC.
Aytaj Mahammadli:
    - Wrote documentation for code and models.
    - Implemented hydra.
Josh Knize:
    - Wrote README.md and PHASE1.md files.
    - Added typing with mypy and code formatting with ruff.
    - Debugged model training and evaluation code
    - Built Docker image
    - Built logging functionality
    - Configuration/experiment managment with Hydra
    - Added unit testing
    - Maintained main.py for Docker runs

## 7. References
- [x] List of datasets, frameworks, and major third-party tools used

1. Python and Anaconda

2. Tensorflow, pandas, numpy, and pillow libraries

3. Flickr8k dataset

4. CNN and LSTM models

5. ruff for code formatting

6. mypy for typing

7. Docker for containerization

8. logging with logging and rich

9. Hydra for configuration/experiment management

---

> **Tip:** Keep this README updated as your project evolves. Link to each phase deliverable and update the architecture diagram as your pipeline matures.
