# Image Caption Generation MLOps Pipeline

## 1. Team Information
- [x] Team Name: Team 2
- [x] Team Members: Aadarsh Kumar Narain (anarian2@depaul.edu), Aytaj Mahammadli (amahamma@depaul.edu), Josh Knize (jknize1@depaul.edu)
- [x] Course & Section: SE 489 - Section 910

## 2. Project Overview
- [x] Brief summary of the project (2-3 sentences)
- This project aims to provide a complete machine learning (ML) solution to address the problem of image caption generation. This repository can be used to deploy a custom solution consisting of a convolutional neural network (CNN) and long short-term memory (LSTM) network in an MLOps pipeline.
- [x] Problem statement and motivation
- The problem of image caption generation is currently of great interest in artificial intelligence (AI) research. Recent advancements in image classification and object detection, as well as natural language processing (NLP) with large language models (LLMs), has opened up opportunities for further research into automatic image caption generation. The ability to do this task can lead to significant progress in other problems such as image searchability, visual question answering (VQA), image understanding, and multi-modal model capabilities. 
- [x] Main objectives
- Our main project objectives are:
    - Deploy an image captioning ML model for end users to interact with
    - Develop a continuously integrated repository of organized and well-documented code and data
    - Use proper version control for our codebase, models, and data
    - Ensure reproducibility across our experiments and repository

## 3. Project Architecture Diagram
- [x] Insert or link to your architecture diagram (e.g., draw.io, PNG, etc.)
- See architecture diagram in docs/architecture.png

## 4. Phase Deliverables
- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions
- [ ] How to set up the environment (conda/pip, requirements.txt, Docker, etc.)
- Clone the Github repository and open the directory in the command line.
- Create a conda environment: 
    - `conda create -n image_caption_env python=3.10`
- Activate the conda environment and install dependencies: 
    - `conda activate image_caption_env`
    - `pip install -r requirements.txt`
- [ ] How to run the code and reproduce results
- Download the data and models from this location: **Insert Google Drive Link**
- Store the models in the models/ directory and the data in the data/ directory.
- Execute the `imageCaption.ipynb` notebook.

## 6. Contribution Summary
- [ ] Briefly describe each team member's contributions
- Aadarsh Narain: Created Git repository and structure. Uploaded models, data, and code. 
- Aytaj Mahammadli: Wrote documentation for code and models.
- Josh Knize: Wrote README.md and PHASE1.md files.

## 7. References
- [x] List of datasets, frameworks, and major third-party tools used
- Python and Anaconda
- Tensorflow, pandas, numpy, and pillow libraries
- Flickr8k dataset
- CNN and LSTM models

---

> **Tip:** Keep this README updated as your project evolves. Link to each phase deliverable and update the architecture diagram as your pipeline matures.