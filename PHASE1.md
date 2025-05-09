# PHASE 1: Project Design & Model Development

## 1. Project Proposal
- [x] **1.1 Project Scope and Objectives**

  - [x] Problem statement

The problem of image caption generation is currently of great interest in artificial intelligence (AI) research. Recent advancements in image classification and object detection, as well as natural language processing (NLP) with large language models (LLMs), has opened up opportunities for further research into automatic image caption generation. The ability to do this task can lead to significant progress in other problems such as image searchability, visual question answering (VQA), image understanding, and multi-modal model capabilities. 

  - [x] Project objectives and expected impact

This project aims to provide a complete machine learning (ML) solution to address the problem of image caption generation. This repository can be used to deploy a custom solution consisting of a convolutional neural network (CNN) and long short-term memory (LSTM) network in an MLOps pipeline. The success and deployment of these models will make it easier for collaborators to engineer and develop solutions to downstream tasks such as the ones mentioned in the problem statement above. 

  - [x] Success metrics

The models will be evaluated on (BiLingual Evaluation Understudy) BLEU score. A BLEU score is a value between 0 and 1 that indicates the quality of the model's text output. This is meant to quantify the correlation from the machine output to a human ground truth, which is stored in an evaluation dataset corresponding to each image in the dataset. The scores are calculated for each sample and averaged over the entire dataset. 

  - [x] 300+ word project description

The aim of our project is to implement a machine learning model for automated image caption generation. This model will consist primarily of two parts: a convolutional neural network (CNN) for image feature extraction, and a long short-term memory network for caption generation. The model will be trained on the Flickr8k image dataset with five different captions associated with each image. The models will be trained using Tensorflow, and we are interested in using Tensorboard to integrate monitoring and evaluation into our final project solution.

The model will be implemented in a machine learning operations (MLOps) pipeline. Our complete solution will include all phases of the ML project lifecycle. In the first part of our project, we create a Github repository to contain our codebase, documentation, environment details, and other necessary files and links. We will use the repository to properly version our code and collaborate among team members. Additionally, we properly document our code with best practices during the first stage of the project. We provide further documentation for our project with a detailed README file containing useful information, project details, a working architecture diagram, and other relevant details. This file is also included to provide specific details for Phase 1 of our project.

In later phases of the project, we will work to make a reproducible solution by using Docker containers. Additionally, we will implement proper version control for both our models and data, and we will create a model registry where we can manage our models. We will ultimately create an MLOps solution that allows continuous integration and delivery of our models. We will explore monitoring strategies so data drift and concept drift could be detected and alerted as new data flows through the model. Finally, we will consider scalability and ethical considerations. 

- [ ] **1.2 Selection of Data**
  - [ ] Dataset(s) chosen and justification
  - [ ] Data source(s) and access method
  - [ ] Preprocessing steps
- [ ] **1.3 Model Considerations**
  - [ ] Model architecture(s) considered
  - [ ] Rationale for model choice
  - [ ] Source/citation for any pre-built models
- [x] **1.4 Open-source Tools**
  - [x] Third-party package(s) selected (not PyTorch or course-used tools)

Tensorflow

  - [x] Brief description of how/why used

Tensorflow is a common, trusted package for machine learning. Because of its popularity, we can easily access support online for it. 

## 2. Code Organization & Setup
- [ ] **2.1 Repository Setup**
  - [x] GitHub repo created
  - [x] Cookiecutter or similar structure used
- [ ] **2.2 Environment Setup**
  - [x] Python virtual environment
  - [ ] requirements.txt or environment.yml
  - [ ] (Optional) Google Colab setup

## 3. Version Control & Collaboration
- [x] **3.1 Git Usage**
  - [x] Regular commits with clear messages
  - [x] Branching and pull requests
- [x] **3.2 Team Collaboration**
  - [x] Roles assigned
  - [x] Code reviews and merge conflict resolution

## 4. Data Handling
- [x] **4.1 Data Preparation**
  - [x] Cleaning, normalization, augmentation scripts
- [ ] **4.2 Data Documentation**
  - [ ] We use DVC (Data Version Control) to manage and version the dataset efficiently. Large files are not stored directly in Git. Instead, we used the following approach:
    - Installed DVC and the Google Drive plugin:  
      `pip install dvc dvc-gdrive`
    - Configured the DVC remote to use a service account:
      - `dvc remote modify myremote gdrive_use_service_account true`
      - `dvc remote modify myremote gdrive_service_account_json_file_path .secrets/dvc-drive-key.json`
    - Pulled the data from the remote with:  
      `dvc pull`
    - The dataset is managed through `.dvc` files tracked by Git, with the data itself stored securely in our linked Google Drive remote.
    - This setup ensures efficient data sharing, reproducibility, and proper access control within the team.

## 5. Model Training
- [ ] **5.1 Training Infrastructure**
  - [ ] Training environment setup (e.g., Colab, GPU)
- [x] **5.2 Initial Training & Evaluation**
  - [x] Baseline model results

The final BLEU-1 score of the model was 0.2117. 

  - [x] Evaluation metrics

The model was evaluated on the BLEU score and cross-entropy loss. 

## 6. Documentation & Reporting
- [x] **6.1 Project README**
  - [x] [Overview, setup, replication steps, dependencies, team contributions](docs/README.md)
- [ ] **6.2 Code Documentation**
  - [ ] Docstrings, inline comments, code style (ruff), type checking (mypy), Makefile docs

---

> **Checklist:** Use this as a guide. Not all items are required, but thorough documentation and reproducibility are expected.
