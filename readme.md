# Proactive Retrieval-based Chatbots based on RelevantKnowledge and Goals

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

#### News

## Abstract
This repository contains the source code and datasets for the SIGIR 2021 paper [Proactive Retrieval-based Chatbots based on Relevant Knowledge and Goals]() by Zhu et al. <br>

A proactive dialogue system has the ability to proactively lead the conversation. Different from the general chatbots which only react to the user, proactive dialogue systems can be used to achieve some goals, e.g., to recommend some items to the user. Background knowledge is essential to enable smooth and natural transitions in dialogue. In this paper, we propose a new multi-task learning framework for retrieval-based knowledge-grounded proactive dialogue. To determine the relevant knowledge to be used, we frame knowledge prediction as a complementary task and use explicit signals to supervise its learning. The final response is selected according to the predicted knowledge, the goal to achieve, and the context. Experimental results show that explicit modeling of knowledge prediction and goal selection can greatly improve the final response selection.  

Authors: Yutao Zhu, Jian-Yun Nie, Kun Zhou, Pan Du, Hao Jiang, Zhicheng Dou

## Requirements
I test the code with the following packages. Other versions may also work, but I'm not sure. <br>
- Python 3.5 <br>
- Pytorch 1.3.1 (with GPU support)<br>

## Usage


## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{ZhuNZDJD21,
  author    = {Yutao Zhu and
               Jian{-}Yun Nie and
               Kun Zhou and
               Pan Du and
               Hao Jiang and
               Zhicheng Dou},
  editor    = {Fernando Diaz and
               Chirag Shah and
               Torsten Suel and
               Pablo Castells and
               Rosie Jones and
               Tetsuya Sakai},
  title     = {Proactive Retrieval-based Chatbots based on Relevant Knowledge and
               Goals},
  booktitle = {{SIGIR} '21: The 44th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Virtual Event, Canada, July
               11-15, 2021},
  pages     = {2000--2004},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3404835.3463011},
  doi       = {10.1145/3404835.3463011},
}
```
