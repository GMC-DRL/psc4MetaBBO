# Useful Papers and Source Codes for Meta Black-Box Optimization (MetaBBO)

This respository aims to maintain a list of useful relevant papers and open source codes for MetaBBO. Our implementations of these methods can be accessed in [MetaBox](https://github.com/GMC-DRL/MetaBox).

- [Useful Papers and Source Codes for Meta Black-Box Optimization (MetaBBO)](#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo)
  - [1. Survey Papers](#1-survey-papers)
  - [2. MetaBBO with Reinforcement Learning](#2-metabbo-with-reinforcement-learning-metabbo-rl)
    - [2.1. Differential Evolution](#21-differential-evolution)
    - [2.2. Partical Swarm Optimization](#22-partical-swarm-optimization)
    - [2.3. Genetic Algorithm](#23-genetic-algorithm)
    - [2.4. Evolution Strategy](#24-evolution-strategy)
    - [2.5. Others](#25-others)
  - [3. MetaBBO with Supervised Learning](#3-metabbo-with-supervised-learning-metabbo-sl)
  - [4. MetaBBO with Self-Referential Search](#4-metabbo-with-self-referential-search-metabbo-sr)

## 1. Survey Papers

|Paper|About|
|:-:|:-:|
|Chernigovskaya, Maria, Andrey Kharitonov, and Klaus Turowski. "[**A Recent Publications Survey on Reinforcement Learning for Selecting Parameters of Meta-Heuristic and Machine Learning Algorithms**](https://www.scitepress.org/Papers/2023/119543/119543.pdf)." CLOSER. 2023.|[PDF](Survey/Chernigovskaya%20et%20al/A%20Recent%20Publications%20Survey%20on%20Reinforcement%20Learning%20for%20Selecting%20Parameters%20of%20Meta-Heuristic%20and%20Machine%20Learning%20Algorithms.pdf)  [BibTex](Survey/Chernigovskaya%20et%20al/BibTex)|
|Drugan, Madalina M. "[**Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms**](https://www.sciencedirect.com/science/article/abs/pii/S2210650217302766)." Swarm and Evolutionary Computation 44 (2019): 228-246.|[PDF](Survey/Drugan%20et%20al/Reinforcement%20learning%20versus%20evolutionary%20computation%20A%20survey%20on%20hybrid%20algorithms.pdf)  [BibTex](Survey/Drugan%20et%20al/BibTex)|

## 2. MetaBBO with Reinforcement Learning (MetaBBO-RL)

### 2.1. Differential Evolution

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|*RLDMDE|Yang, Qingyong, et al. "[**Dynamic multi-strategy integrated differential evolution algorithm based on reinforcement learning for optimization problems**](https://link.springer.com/article/10.1007/s40747-023-01243-9)." Complex & Intelligent Systems (2023): 1-33.|-|[PDF](MetaBBO-RL/Differential%20Evolution/RLDMDE/Dynamic%20multi-strategy%20integrated%20differential%20evolution%20algorithm%20based%20on%20reinforcement%20learning%20for%20optimization%20problems.pdf)  [BibTex](MetaBBO-RL/Differential%20Evolution/RLDMDE/BibTex)|
|RL-HPSDE|Tan, Zhiping, et al. "[**Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning**](https://www.sciencedirect.com/science/article/pii/S2210650222001602)." Swarm and Evolutionary Computation 75 (2022): 101194.|-|[PDF](MetaBBO-RL/Differential%20Evolution/RL-HPSDE/Differential%20evolution%20with%20hybrid%20parameters%20and%20mutation%20strategies%20based%20on%20reinforcement%20learning.pdf)   [BibTex](MetaBBO-RL/Differential%20Evolution/RL-HPSDE/BibTex)|
|DE-DQN|Tan, Zhiping, and Kangshun Li. "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing 111 (2021): 107678.|-|[PDF](MetaBBO-RL/Differential%20Evolution/DE-DQN/Differential%20evolution%20with%20mixed%20mutation%20strategy%20based%20on%20deep%20reinforcement%20learning.pdf)  [BibTex](MetaBBO-RL/Differential%20Evolution/DE-DQN/BibTex)|
|LDE|Sun, Jianyong, et al. "[**Learning Adaptive Differential Evolution Algorithm from Optimization Experiences by Policy Gradient**](https://ieeexplore.ieee.org/abstract/document/9359652)." IEEE Transactions on Evolutionary Computation 25.4 (2021): 666-680.|[yierh/LDE](https://github.com/yierh/LDE)|[PDF](MetaBBO-RL/Differential%20Evolution/LDE/Learning%20Adaptive%20Differential%20Evolution%20Algorithm%20from%20Optimization%20Experiences%20by%20Policy%20Gradient.pdf)  [BibTex](MetaBBO-RL/Differential%20Evolution/LDE/BibTex)|
|*MARLwCMA|Sallam, Karam M., et al. "[**Evolutionary framework with reinforcement learning-based mutation adaptation**](https://ieeexplore.ieee.org/abstract/document/9239320/)." IEEE Access 8 (2020): 194045-194071.|-|[PDF](MetaBBO-RL/Differential%20Evolution/MARLwCMA/Evolutionary%20framework%20with%20reinforcement%20learning-based%20mutation%20adaptation.pdf)  [BibTex](MetaBBO-RL/Differential%20Evolution/MARLwCMA/BibTex)|
|DE-DDQN|Sharma, Mudita, et al. "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference. 2019.|[mudita11/DE-DDQN](https://github.com/mudita11/DE-DDQN)|[PDF](MetaBBO-RL/Differential%20Evolution/DE-DDQN/Deep%20reinforcement%20learning%20based%20parameter%20control%20in%20differential%20evolution.pdf)  [BibTex](MetaBBO-RL/Differential%20Evolution/DE-DDQN/BibTex)|
|*DE-RLFR|Li, Zhihui, et al. "[**Differential evolution based on reinforcement learning with fitness ranking for solving multimodal multiobjective problems**](https://www.sciencedirect.com/science/article/pii/S2210650218310575)." Swarm and Evolutionary Computation 49 (2019): 234-244.|-|[PDF](MetaBBO-RL/Differential%20Evolution/DE-RLFR/Differential%20evolution%20based%20on%20reinforcement%20learning%20with%20fitness%20ranking%20for%20solving%20multimodal%20multiobjective%20problems.pdf)  [BibTex](MetaBBO-RL/Differential%20Evolution/DE-RLFR/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.2. Partical Swarm Optimization

### 2.3. Genetic Algorithm

### 2.4. Evolution Strategy

### 2.5. Others

## 3. MetaBBO with Supervised Learning (MetaBBO-SL)

## 4. MetaBBO with Self-Referential Search (MetaBBO-SR)
