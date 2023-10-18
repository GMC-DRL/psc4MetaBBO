# Useful Papers and Source Codes for Meta Black-Box Optimization (MetaBBO)

This respository aims to maintain a list of useful relevant papers and open source codes for MetaBBO.

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

Chernigovskaya, Maria, Andrey Kharitonov, and Klaus Turowski. "[**A Recent Publications Survey on Reinforcement Learning for Selecting Parameters of Meta-Heuristic and Machine Learning Algorithms**](https://www.scitepress.org/Papers/2023/119543/119543.pdf)." CLOSER. 2023.
[BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:4jdFv92JE6UJ:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl4nv_Nw:AFWwaeYAAAAAZS_p5NzkTxf6faz97te3d01HrFs&scisig=AFWwaeYAAAAAZS_p5MLs7wpA0fzU8b3nXpq9DHw&scisf=4&ct=citation&cd=-1&hl=en)

Drugan, Madalina M. "[**Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms**](https://www.researchgate.net/profile/Mohamed-Mourad-Lafifi/post/Any_reference_paper_with_detailing_on_how_to_hybradize_Q-Learning_algorithm_with_Genetic_Algorithm_especially_for_a_variable_optimization_problem/attachment/60213347e501d80001b17f3e/AS%3A988921009819663%401612788550903/download/Reinforcement+learning+versus+evolutionary+computation+_+A+survey+on+hybrid+algorithms+_+Drugan2018.pdf)." Swarm and Evolutionary Computation 44 (2019): 228-246. [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:ubDobEqzKFQJ:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl4ntLtM:AFWwaeYAAAAAZS_rNtONrncGq42OsZYGX_5sO2c&scisig=AFWwaeYAAAAAZS_rNunOHkUDoGlMUbV702WOFEE&scisf=4&ct=citation&cd=-1&hl=en)

## 2. MetaBBO with Reinforcement Learning (MetaBBO-RL)

### 2.1. Differential Evolution

|Algorithm|Paper|Original Repository|
|:-:|:-:|:-:|
|RLDMDE|Yang, Qingyong, et al. "[**Dynamic multi-strategy integrated differential evolution algorithm based on reinforcement learning for optimization problems**](https://link.springer.com/article/10.1007/s40747-023-01243-9)." Complex & Intelligent Systems (2023): 1-33.  [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:G5HPRFAvkuoJ:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl4n6BS8:AFWwaeYAAAAAZS_8HS-L19cKZvppuxG22uN4dJs&scisig=AFWwaeYAAAAAZS_8He7BeQppwwQV6ewudr2nK4E&scisf=4&ct=citation&cd=-1&hl=en)|-|
|RL-HPSDE|Tan, Zhiping, et al. "[**Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning**](https://www.sciencedirect.com/science/article/pii/S2210650222001602)." Swarm and Evolutionary Computation 75 (2022): 101194.  [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:ZDaX_JCwM18J:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl4n7TuI:AFWwaeYAAAAAZS_9VuKRYblgisoNwfYtz8IXs4Y&scisig=AFWwaeYAAAAAZS_9Vk7GnXn-4YUbfSmby2w_N-s&scisf=4&ct=citation&cd=-1&hl=en)|-|
|DE-DQN|Tan, Zhiping, and Kangshun Li. "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing 111 (2021): 107678.  [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:FT9DB-IsY28J:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl4n0_XE:AFWwaeYAAAAAZS_y5XH_TfEdp24uinUAtgrAoc0&scisig=AFWwaeYAAAAAZS_y5U91OuZBUuZkW0ZTGDSPj58&scisf=4&ct=citation&cd=-1&hl=en)|-|
|LDE|Sun, Jianyong, et al. "[**Learning Adaptive Differential Evolution Algorithm from Optimization Experiences by Policy Gradient**](https://arxiv.org/pdf/2102.03572)." IEEE Transactions on Evolutionary Computation 25.4 (2021): 666-680.  [BibTex](MetaBBO-RL/LDE/BibTex)|[yierh/LDE](https://github.com/yierh/LDE)|
|MARLwCMA|Sallam, Karam M., et al. "[**Evolutionary framework with reinforcement learning-based mutation adaptation**](https://ieeexplore.ieee.org/iel7/6287639/6514899/09239320.pdf)." IEEE Access 8 (2020): 194045-194071.  [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VMrP85q_bBgJ:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl5YGPc8:AFWwaeYAAAAAZTAAJc9qHQImHyOy4Vl2mRgHhic&scisig=AFWwaeYAAAAAZTAAJWUXLnRSDGiSuWoZlL1gNjc&scisf=4&ct=citation&cd=-1&hl=en)|-|
|DE-DDQN|Sharma, Mudita, et al. "[**Deep reinforcement learning based parameter control in differential evolution**](https://arxiv.org/pdf/1905.08006)." Proceedings of the Genetic and Evolutionary Computation Conference. 2019.  [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:coilmqXt4_IJ:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl4n1bWQ:AFWwaeYAAAAAZS_zdWQ7JQQ9CrsrzMwyqURwAzM&scisig=AFWwaeYAAAAAZS_zdTs2V4eGjWkK-aMX0IEg8Iw&scisf=4&ct=citation&cd=-1&hl=en)|[mudita11/DE-DDQN](https://github.com/mudita11/DE-DDQN)|
|DE-RLFR|Li, Zhihui, et al. "[**Differential evolution based on reinforcement learning with fitness ranking for solving multimodal multiobjective problems**](https://www.sciencedirect.com/science/article/pii/S2210650218310575)." Swarm and Evolutionary Computation 49 (2019): 234-244.  [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:NapbmYYF1R8J:scholar.google.com/&output=citation&scisdr=ClFw6mjsEOXkl4n4jUg:AFWwaeYAAAAAZS_-lUjwjdqg0EG5HMGJ5BIZ8yE&scisig=AFWwaeYAAAAAZS_-lTQkIA1D_KsWdpAPr91Tfs0&scisf=4&ct=citation&cd=-1&hl=en)|-|


### 2.2. Partical Swarm Optimization

### 2.3. Genetic Algorithm

### 2.4. Evolution Strategy

### 2.5. Others

## 3. MetaBBO with Supervised Learning (MetaBBO-SL)

## 4. MetaBBO with Self-Referential Search (MetaBBO-SR)
