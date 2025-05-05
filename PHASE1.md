# PHASE 1: Project Design & Model Development

## 1. Project Proposal
- [ ] **1.1 Project Scope and Objectives**
  - [x] Problem statement
  - The problem of image caption generation is currently of great interest in artificial intelligence (AI) research. Recent advancements in image classification and object detection, as well as natural language processing (NLP) with large language models (LLMs), has opened up opportunities for further research into automatic image caption generation. The ability to do this task can lead to significant progress in other problems such as image searchability, visual question answering (VQA), image understanding, and multi-modal model capabilities. 
  - [x] Project objectives and expected impact
  - This project aims to provide a complete machine learning (ML) solution to address the problem of image caption generation. This repository can be used to deploy a custom solution consisting of a convolutional neural network (CNN) and long short-term memory (LSTM) network in an MLOps pipeline. The success and deployment of these models will make it easier for collaborators to engineer and develop solutions to downstream tasks such as the ones mentioned in the problem statement above. 
  - [x] Success metrics
  - The models will be evaluated on (BiLingual Evaluation Understudy) BLEU score. A BLEU score is a value between 0 and 1 that indicates the quality of the model's text output. This is meant to quantify the correlation from the machine output to a human ground truth, which is stored in an evaluation dataset corresponding to each image in the dataset. The scores are calculated for each sample and averaged over the entire dataset. 
  - [ ] 300+ word project description
- [ ] **1.2 Selection of Data**
  - [ ] Dataset(s) chosen and justification
  - [ ] Data source(s) and access method
  - [ ] Preprocessing steps
- [ ] **1.3 Model Considerations**
  - [ ] Model architecture(s) considered
  - [ ] Rationale for model choice
  - [ ] Source/citation for any pre-built models
- [ ] **1.4 Open-source Tools**
  - [ ] Third-party package(s) selected (not PyTorch or course-used tools)
  - [ ] Brief description of how/why used

## 2. Code Organization & Setup
- [ ] **2.1 Repository Setup**
  - [x] GitHub repo created
  - [ ] Cookiecutter or similar structure used
- [ ] **2.2 Environment Setup**
  - [ ] Python virtual environment
  - [ ] requirements.txt or environment.yml
  - [ ] (Optional) Google Colab setup

## 3. Version Control & Collaboration
- [ ] **3.1 Git Usage**
  - [ ] Regular commits with clear messages
  - [ ] Branching and pull requests
- [ ] **3.2 Team Collaboration**
  - [ ] Roles assigned
  - [ ] Code reviews and merge conflict resolution

## 4. Data Handling
- [ ] **4.1 Data Preparation**
  - [ ] Cleaning, normalization, augmentation scripts
- [ ] **4.2 Data Documentation**
  - [ ] Description of data prep process

## 5. Model Training
- [ ] **5.1 Training Infrastructure**
  - [ ] Training environment setup (e.g., Colab, GPU)
- [ ] **5.2 Initial Training & Evaluation**
  - [ ] Baseline model results
  - [ ] Evaluation metrics

## 6. Documentation & Reporting
- [ ] **6.1 Project README**
  - [ ] Overview, setup, replication steps, dependencies, team contributions
- [ ] **6.2 Code Documentation**
  - [ ] Docstrings, inline comments, code style (ruff), type checking (mypy), Makefile docs

---

> **Checklist:** Use this as a guide. Not all items are required, but thorough documentation and reproducibility are expected.
