# Useful Papers and Source Codes for Meta Black-Box Optimization (MetaBBO)

This respository aims to maintain a list of useful relevant papers and open source codes for MetaBBO. Our implementations of some of these methods can be accessed in [MetaBox](https://github.com/GMC-DRL/MetaBox).

- [1. Survey Papers \& Benchmarks](#1-survey-papers--benchmarks)
  - [1.1. Survey Papers](#11-survey-papers)
  - [1.2. Benchmarks](#12-benchmarks)
- [2. MetaBBO](#2-metabbo)
  - [2.1. MetaBBO-RL](#21-metabbo-with-reinforcement-learning-metabbo-rl)
    - [2.1.1. Algorithm Selection](#211-operator-selection)
    - [2.1.2. Algorithm Configuration](#212-parameter-contorl)
    - [2.1.3. Algorithm Generation \& Parameter](#213-operator--parameter)
    - [2.1.4. Algorithm Imitation](#214-symbolic)
    - [2.1.5. Others](#215-others)
  - [2.2. MetaBBO-SL](#22-metabbo-with-supervised-learning-metabbo-sl)
    - [2.2.1. Algorithm Selection](#221-operator-selection)
    - [2.2.2. Algorithm Configuration](#222-parameter-contorl)
    - [2.2.3. Algorithm Generation \& Parameter](#223-operator--parameter)
    - [2.2.4. Algorithm Imitation](#224-symbolic)
    - [2.2.5. Others](#225-others)
  - [2.3. MetaBBO-NE](#23-metabbo-with-neuroevolution-metabbo-ne)
    - [2.3.1. Algorithm Selection](#231-operator-selection)
    - [2.3.2. Algorithm Configuration](#232-parameter-contorl)
    - [2.3.3. Algorithm Generation \& Parameter](#233-operator--parameter)
    - [2.3.4. Algorithm Imitation](#234-symbolic)
    - [2.3.5. Others](#235-others)
  - [2.4. MetaBBO-ICL](#24-metabbo-with-in-context-learning )
    - [2.4.1. Algorithm Selection](#241-operator-selection)
    - [2.4.2. Algorithm Configuration](#242-parameter-contorl)
    - [2.4.3. Algorithm Generation \& Parameter](#243-operator--parameter)
    - [2.4.4. Algorithm Imitation](#244-symbolic)
    - [2.4.5. Others](#245-others)
  - [2.5. Others](#25-others)
    - [2.5.1 Evaluation Indicator](#251-evaluation-indicator)
    - [2.5.2 Landscape Feature](#252-landscape-feature)
- [3. Classic BBO](#3-classic-bbo)
  - [3.1. Differential Evolution](#31-differential-evolution)
  - [3.2. Partical Swarm Optimization](#32-partical-swarm-optimization)
  - [3.3. Evolution Strategies](#33-evolution-strategies)
  - [3.4. Bayesian Optimization](#34-bayesian-optimization)
  - [3.5. Others](#35-others)


## 1. Survey Papers \& Benchmarks

### 1.1. Survey Papers

|Paper|
|:-:|
|Li P, Hao J, Tang H, et al. "[**Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey on Hybrid Algorithms**](https://ieeexplore.ieee.org/abstract/document/10637292). IEEE Transactions on Evolutionary Computation. (2024).
|Song Y, Wu Y, Guo Y, et al. "[**Reinforcement learning-assisted evolutionary algorithm: A survey and research opportunities**](https://www.sciencedirect.com/science/article/pii/S2210650224000506). Swarm and Evolutionary Computation. (2024).
|Nikolikj, Ana, et al. "[**Quantifying Individual and Joint Module Impact in Modular Optimization Frameworks**](https://arxiv.org/abs/2405.11964)." 2024 IEEE Congress on Evolutionary Computation (CEC). (2024).
|Qian, Chao, Ke Xue, and Ren-Jian Wang. "[**Quality-Diversity Algorithms Can Provably Be Helpful for Optimization**](https://arxiv.org/abs/2401.10539)." arXiv preprint arXiv:2401.10539. (2024).
|Huang, Beichen, et al. "[**Exploring the True Potential: Evaluating the Black-box Optimization Capability of Large Language Models**](https://arxiv.org/abs/2404.06290)." arXiv preprint arXiv:2404.06290. (2024).
|Chernigovskaya, Maria, Andrey Kharitonov, and Klaus Turowski. "[**A Recent Publications Survey on Reinforcement Learning for Selecting Parameters of Meta-Heuristic and Machine Learning Algorithms**](https://www.scitepress.org/Papers/2023/119543/119543.pdf)." CLOSER. (2023).
|Drugan, Madalina M. "[**Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms**](https://www.sciencedirect.com/science/article/abs/pii/S2210650217302766)." Swarm and Evolutionary Computation. (2019).

### 1.2. Benchmarks

|Benchmark|Paper|Original Repository|Optimization Type|
|:-:|:-:|:-:|:-:|
|GP-based|He Y, Aranha C. "[**Evolving Benchmark Functions to Compare Evolutionary Algorithms via Genetic Programming**](https://arxiv.org/abs/2403.14146)". arXiv preprint arXiv:2403.14146 (2024).|[GP-based](https://github.com/Y1fanHE/cec2024)||
|SELECTOR|Benjamins, Carolin, et al. "[**Instance Selection for Dynamic Algorithm Configuration with Reinforcement Learning: Improving Generalization**](https://arxiv.org/abs/2407.13513)." arXiv preprint arXiv:2407.13513 (2024).|[automl/instance-dac]( https://github.com/automl/instance-dac)||
|MetaBox|Ma, Zeyuan, et al. "[**MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning**](https://neurips.cc/virtual/2023/oral/73737)." Advances in Neural Information Processing Systems 36 (2023).|[GMC-DRL/MetaBox]( https://github.com/GMC-DRL/MetaBox)||
|NN-based|Prager R P, Dietrich K, Schneider L, et al. "[**Neural Networks as Black-Box Benchmark Functions Optimized for Exploratory Landscape Features**](https://dl.acm.org/doi/abs/10.1145/3594805.3607136)" Proceedings of the 17th ACM/SIGEVO Conference on Foundations of Genetic Algorithms (2023).| - | |
|NeuroEvoBench|Lange, Robert, Yujin Tang, and Yingtao Tian. "[**Neuroevobench: Benchmarking evolutionary optimizers for deep learning applications**](https://neurips.cc/virtual/2023/oral/73737)." Advances in Neural Information Processing Systems 36 (2023)|[neuroevobench/neuroevobench](https://github.com/neuroevobench/neuroevobench)||
|MA-BBOB|Vermetten, Diederick, et al. "[**MA-BBOB: A Problem Generator for Black-Box Optimization Using Affine Combinations and Shifts**](https://arxiv.org/abs/2312.11083)." arXiv preprint arXiv:2312.11083 (2023).|[Dvermetten/Many-affine-BBOB](https://github.com/Dvermetten/Many-affine-BBOB)||
|IEEE CEC 2022|Abhishek Kumar, Kenneth V. Price, Ali Wagdy Mohamed, Anas A. Hadi, P. N. Suganthan, "[**Problem definitions and evaluation criteria for the cec 2022 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2022/CEC2022.htm)." Technical Report 2022|[P-N-Suganthan/2022-SO-BO](https://github.com/P-N-Suganthan/2022-SO-BO)||
|Affine Recombination|Dietrich K, Mersmann O. "[**Increasing the diversity of benchmark function sets through affine recombination**](https://link.springer.com/chapter/10.1007/978-3-031-14714-2_41)" International Conference on Parallel Problem Solving from Nature. (2022).| - | |
|IEEE CEC 2021|Ali Wagdy, Anas A Hadi, Ali K. Mohamed, Prachi Agrawal, Abhishek Kumar and P. N. Suganthan, "[**Problem definitions and evaluation criteria for the cec 2021 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm)." Technical Report 2021|[P-N-Suganthan/2021-SO-BCO](https://github.com/P-N-Suganthan/2021-SO-BCO)||
|Zigzag BBO|Kudela, Jakub. "[**Novel zigzag-based benchmark functions for bound constrained single objective optimization**](https://ieeexplore.ieee.org/abstract/document/9504720/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, (2021).<br>Kudela, Jakub, and Radomil Matousek. "[**New benchmark functions for single-objective optimization based on a zigzag pattern**](https://ieeexplore.ieee.org/abstract/document/9684455/)." IEEE Access 10 (2022).|[JakubKudela89/Zigzag](https://github.com/JakubKudela89/Zigzag)||
|HPOBench|Eggensperger, Katharina, et al. "[**HPOBench: A collection of reproducible multi-fidelity benchmark problems for HPO**](https://arxiv.org/abs/2109.06716)." arXiv preprint arXiv:2109.06716 (2021).|[automl/HPOBench](https://github.com/automl/HPOBench)||
|DACBench|Eimer, Theresa, et al. "[**DACBench: A benchmark library for dynamic algorithm configuration**](https://arxiv.org/abs/2105.08541)." arXiv preprint arXiv:2105.08541 (2021).|[automl/DACBench](https://github.com/automl/DACBench)||
|Olympus|Häse, Florian, et al. "[**Olympus: a benchmarking framework for noisy optimization and experiment planning**](https://iopscience.iop.org/article/10.1088/2632-2153/abedc8/meta)." Machine Learning: Science and Technology (2021).|[aspuru-guzik-group/olympus](https://github.com/aspuru-guzik-group/olympus)||
|NeurIPS BBO challenge|Turner R, Eriksson D, McCourt M, et al. "[**Bayesian optimization is superior to random search for machine learning hyperparameter tuning: Analysis of the black-box optimization challenge 2020**](https://proceedings.mlr.press/v133/turner21a.html)" NeurIPS 2020 Competition and Demonstration Track. (2021)|[NeurIPS BBO challenge](https://github.com/rdturnermtl/bbo_challenge_starter_kit/) | |
|Random function generator|Tian Y, Peng S, Zhang X, et al. "[**A recommender system for metaheuristic algorithms for continuous optimization based on deep recurrent neural networks**](https://ieeexplore.ieee.org/abstract/document/9187549)". IEEE transactions on artificial intelligence (2020).|[Random function generator](https://github.com/BIMK/Algorithm-Recommendation) | |
|CEC 2020 competition on real-world optimization problem|Kumar A, Wu G, Ali M Z, et al. "[**A test-suite of non-convex constrained optimization problems from the real-world and some baseline results**](https://www.sciencedirect.com/science/article/pii/S2210650219308946). Swarm and Evolutionary Computation (2020).|[CEC 2020 real-world](https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation)||
|COCO|Hansen, Nikolaus, et al. "[**COCO: A platform for comparing continuous optimizers in a black-box setting**](https://www.tandfonline.com/doi/abs/10.1080/10556788.2020.1808977)." Optimization Methods and Software (2021).|[numbbo/coco](https://github.com/numbbo/coco)||
|EVOBBO|Muñoz, Mario A., and Kate Smith-Miles. "[**Generating new space-filling test instances for continuous black-box optimization**](https://direct.mit.edu/evco/article-abstract/28/3/379/94997)." Evolutionary computation (2020).|[andremun/EVOBBO_Instances](https://github.com/andremun/EVOBBO_Instances)||
|Bayesmark|Turner R, Eriksson D. "[**Bayesmark: Benchmark framework to easily compare bayesian optimization methods on real machine learning tasks**](https://bayesmark.readthedocs.io/en/latest/)." (2019). |[Bayesmark](https://github. com/uber/bayesmark)| |
|IOHprofiler (IOHexperimenter)|Doerr, Carola, et al. "[**IOHprofiler: A benchmarking and profiling tool for iterative optimization heuristics**](https://arxiv.org/abs/1810.05281)." arXiv preprint arXiv:1810.05281 (2018).<br>de Nobel, Jacob, et al. "[**Iohexperimenter: Benchmarking platform for iterative optimization heuristics**](https://direct.mit.edu/evco/article/doi/10.1162/evco_a_00342/116949)." Evolutionary Computation (2023): 1-6.|[IOHprofiler/<br>IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter)||
|MTMOOP|Yuan Y, Ong Y S, Feng L, et al. "[**Evolutionary multitasking for multiobjective continuous optimization: Benchmark problems, performance metrics and baseline results**](https://arxiv.org/abs/1706.02766)." arXiv preprint arXiv:1706.02766 (2017).|- | |
|MTSOP|Da B, Ong Y S, Feng L, et al. "[**Evolutionary multitasking for single-objective continuous optimization: Benchmark problems, performance metric, and baseline results**](https://arxiv.org/abs/1706.03470)". arXiv preprint arXiv:1706.03470 (2017).|- | |
|IEEE CEC 2017|N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu and P. N. Suganthan, "[**Problem definitions and evaluation criteria for the CEC 2017 competition on constrained real-parameter optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)." Technical Report (2017)|[P-N-Suganthan/CEC2017-BoundContrained](https://github.com/P-N-Suganthan/CEC2017-BoundContrained)||
|IEEE CEC 2015|J. J. Liang, B. Y. Qu, P. N. Suganthan, Q. Chen, "[**Problem Definitions and Evaluation Criteria for the CEC 2015 Competition on Learning-based Real-Parameter Single Objective Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2015/CEC2015.htm)", Technical Report, Computational Intelligence Laboratory (2015).|[P-N-Suganthan/CEC2015-Learning-Based](https://github.com/P-N-Suganthan/CEC2015-Learning-Based)||
|AClib|Hutter, Frank, et al. "[**AClib: A benchmark library for algorithm configuration**](https://link.springer.com/chapter/10.1007/978-3-319-09584-4_4)." Learning and Intelligent Optimization: 8th International Conference (2014).|[aclib.net](https://www.aclib.net/)||
|IEEE CEC 2013|J. J. Liang, B-Y. Qu, P. N. Suganthan, Alfredo G. Hernández-Díaz, "[**Problem Definitions and Evaluation Criteria for the CEC 2013 Special Session and Competition on Real-Parameter Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2013/CEC2013.htm)", Technical Report, Computational Intelligence Laboratory (2013).|[P-N-Suganthan/CEC2013](https://github.com/P-N-Suganthan/CEC2013)||
|Protein–Docking|Hwang, Howook, et al. "[**Protein–protein docking benchmark version 4.0**](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22830)." Proteins: Structure, Function, and Bioinformatics (2010).|[Protein–Docking](http://zlab.umassmed.edu/benchmark/)||
|BBOB 2009|Hansen N, Finck S, Ros R, et al. "[**Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions**](https://inria.hal.science/inria-00362633/)". INRIA. (2009). |[BBOB 2009](https://web.archive.org/web/20200811021008/https://coco.gforge.inria.fr/doku.php?id=bbob-2009-results) | |
|WFG|Huband S, Hingston P, Barone L, et al. "[**A review of multiobjective test problems and a scalable test problem toolkit**](https://ieeexplore.ieee.org/abstract/document/1705400)." IEEE Transactions on Evolutionary Computation. (2006).|[WFG](https://github.com/White-Chen/MOEA-Benchmark) ||
|DTLZ|Deb K, Thiele L, Laumanns M, et al. "[**Scalable multi-objective optimization test problems**](https://ieeexplore.ieee.org/abstract/document/1007032)." Proceedings of the 2002 Congress on Evolutionary Computation (2002).|[DTLZ](https://github.com/msu-coinlab/pymop/tree/master?tab=readme-ov-file) ||
|ZDT|Zitzler, E., Deb, K., and Thiele, L. "[**Comparison of Multiobjective Evolutionary Algorithms: Empirical Results**]( https://dl.acm.org/doi/10.1162/106365600568202)." Evolutionary Computation (2000). |[ZDT](https://github.com/White-Chen/MOEA-Benchmark)| |

**The complete list of IEEE CEC series can be access at [ntu.edu.sg](https://www3.ntu.edu.sg/home/epnsugan/index_files/).*

**The complete list of BBOB series can be access at [numbbo](https://numbbo.github.io/workshops/bbob2023.html).*

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>




## 2. MetaBBO

### 2.1. MetaBBO with Reinforcement Learning (MetaBBO-RL)

#### 2.1.1. Operator Selection

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|MRL-MOEA|Wang, Jing, et al. "[**A Novel Multi-State Reinforcement Learning-Based Multi-Objective Evolutionary Algorithm**](https://www.sciencedirect.com/science/article/pii/S0020025524013112)." Information Sciences (2024): 121397.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/MRL-MOEA/A%20Novel%20Multi-State%20Reinforcement%20Learning-Based%20Multi-Objective%20Evolutionary%20Algorithm.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/MRL-MOEA/BibTex)|
|RL-DAS|Guo, Hongshu, et al. "[**Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution**](https://ieeexplore.ieee.org/abstract/document/10496708/)." IEEE Transactions on Systems, Man, and Cybernetics: Systems (2024).|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/RL-DAS/Deep%20Reinforcement%20Learning%20for%20Dynamic%20Algorithm%20Selection%20A%20Proof-of-Principle%20Study%20on%20Differential%20Evolution.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/RL-DAS/BibTex)|
|RLEMMO|Lian, Hongqiao, et al. "[**RLEMMO: Evolutionary Multimodal Optimization Assisted By Deep Reinforcement Learning**](https://arxiv.org/abs/2404.08242)." arXiv preprint arXiv:2404.08242 (2024).|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/RLEMMO/RLEMMO%20Evolutionary%20Multimodal%20Optimization%20Assisted%20By%20Deep%20Reinforcement%20Learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/RLEMMO/BibTex)|
|SA-DQN-DE|Liao, Zuowen, Qishuo Pang, and Qiong Gu. "[**Differential evolution based on strategy adaptation and deep reinforcement learning for multimodal optimization problems**](https://www.sciencedirect.com/science/article/pii/S2210650224001068)." Swarm and Evolutionary Computation 87 (2024): 101568.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/SA-DQN-DE/Differential%20evolution%20based%20on%20strategy%20adaptation%20and%20deep%20reinforcement%20learning%20for%20multimodal%20optimization%20problems.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/SA-DQN-DE/BibTex)|
|PG-DE \& PG-MPEDE|Zhang, Haotian, et al. "[**Learning to select the recombination operator for derivative-free optimization**](https://link.springer.com/article/10.1007/s11425-023-2252-9)." Science China Mathematics (2024): 1-24.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/PGDE&PGMPEDE/Learning%20to%20select%20the%20recombination%20operator%20for%20derivative-free%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/PGDE&PGMPEDE/BibTex)|
|CEDE-DRL|Hu, Zhenzhen, et al. "[**Deep reinforcement learning assisted co-evolutionary differential evolution for constrained optimization**](https://www.sciencedirect.com/science/article/pii/S2210650223001608)." Swarm and Evolutionary Computation 83 (2023): 101387.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/CEDE-DRL/Deep%20reinforcement%20learning%20assisted%20co-evolutionary%20differential%20evolution%20for%20constrained%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/CEDE-DRL/BibTex)|
|RLMMDE|Han, Yupeng, et al. "[**Multi-strategy multi-objective differential evolutionary algorithm with reinforcement learning**](https://www.sciencedirect.com/science/article/pii/S0950705123005518)." Knowledge-Based Systems 277 (2023): 110801.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/RLMMDE/Multi-strategy%20multi-objective%20differential%20evolutionary%20algorithm%20with%20reinforcement%20learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/RLMMDE/BibTex)|
|RLDMDE|Yang, Qingyong, et al. "[**Dynamic multi-strategy integrated differential evolution algorithm based on reinforcement learning for optimization problems**](https://link.springer.com/article/10.1007/s40747-023-01243-9)." Complex & Intelligent Systems (2023): 1-33.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/RLDMDE/Dynamic%20multi-strategy%20integrated%20differential%20evolution%20algorithm%20based%20on%20reinforcement%20learning%20for%20optimization%20problems.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/RLDMDE/BibTex)|
|MPSORL|Meng, Xiaoding, Hecheng Li, and Anshan Chen. "[**Multi-strategy self-learning particle swarm optimization algorithm based on reinforcement learning**](http://www.aimspress.com/aimspress-data/mbe/2023/5/PDF/mbe-20-05-373.pdf)." Mathematical Biosciences and Engineering 20.5 (2023): 8498-8530.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/MPSORL/Multi-strategy%20self-learning%20particle%20swarm%20optimization%20algorithm%20based%20on%20reinforcement%20learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/MPSORL/BibTex)|
|MOEA/D-DQN|Tian, Ye, et al. "[**Deep reinforcement learning based adaptive operator selection for evolutionary multi-objective optimization**](https://ieeexplore.ieee.org/abstract/document/9712324/)." IEEE Transactions on Emerging Topics in Computational Intelligence (2022).|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/MOEA-D-DQN/Deep%20reinforcement%20learning%20based%20adaptive%20operator%20selection%20for%20evolutionary%20multi-objective%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/MOEA-D-DQN/BibTex)|
|DE-DQN|Tan, Zhiping, and Kangshun Li. "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing 111 (2021): 107678.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/DE-DQN/Differential%20evolution%20with%20mixed%20mutation%20strategy%20based%20on%20deep%20reinforcement%20learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/DE-DQN/BibTex)|
|MARLwCMA|Sallam, Karam M., et al. "[**Evolutionary framework with reinforcement learning-based mutation adaptation**](https://ieeexplore.ieee.org/abstract/document/9239320/)." IEEE Access 8 (2020): 194045-194071.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/MARLwCMA/Evolutionary%20framework%20with%20reinforcement%20learning-based%20mutation%20adaptation.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/MARLwCMA/BibTex)|
|DE-DDQN|Sharma, Mudita, et al. "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference. 2019.|[mudita11/DE-DDQN](https://github.com/mudita11/DE-DDQN)|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/DE-DDQN/Deep%20reinforcement%20learning%20based%20parameter%20control%20in%20differential%20evolution.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/DE-DDQN/BibTex)|
|DE-RLFR|Li, Zhihui, et al. "[**Differential evolution based on reinforcement learning with fitness ranking for solving multimodal multiobjective problems**](https://www.sciencedirect.com/science/article/pii/S2210650218310575)." Swarm and Evolutionary Computation 49 (2019): 234-244.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/DE-RLFR/Differential%20evolution%20based%20on%20reinforcement%20learning%20with%20fitness%20ranking%20for%20solving%20multimodal%20multiobjective%20problems.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/DE-RLFR/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

#### 2.1.2. Parameter Contorl

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|GLEET|Ma, Zeyuan, et al. "[**Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning**](https://arxiv.org/abs/2404.08239)." arXiv preprint arXiv:2404.08239 (2024).|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/GLEET/Auto-configuring%20Exploration-Exploitation%20Tradeoff%20in%20Evolutionary%20Computation%20via%20Deep%20Reinforcement%20Learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/GLEET/BibTex)|
|RLMODE|Yu, Xiaobing, et al. "[**Reinforcement learning-based differential evolution algorithm for constrained multi-objective optimization problems**](https://www.sciencedirect.com/science/article/pii/S0952197623020018)." Engineering Applications of Artificial Intelligence 131 (2024): 107817.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLMODE/Reinforcement%20learning-based%20differential%20evolution%20algorithm%20for%20constrained%20multi-objective%20optimization%20problems.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLMODE/BibTex)|
|RLNS|Hong, Jiale, Bo Shen, and Anqi Pan. "[**A reinforcement learning-based neighborhood search operator for multi-modal optimization and its applications**](https://www.sciencedirect.com/science/article/pii/S0957417424000150)." Expert Systems with Applications 246 (2024): 123150.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLNS/A%20reinforcement%20learning-based%20neighborhood%20search%20operator%20for%20multi-modal%20optimization%20and%20its%20applications.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLNS/BibTex)|
|NRLPSO|Li, Wei, et al. "[**Reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy**](https://www.sciencedirect.com/science/article/pii/S2210650223000482)." Swarm and Evolutionary Computation 78 (2023): 101274.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/NRLPSO/Reinforcement%20learning-based%20particle%20swarm%20optimization%20with%20neighborhood%20differential%20mutation%20strategy.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/NRLPSO/BibTex)|
|MOEADRL|Gao, Mengqi, et al. "[**An efficient evolutionary algorithm based on deep reinforcement learning for large-scale sparse multiobjective optimization**](https://link.springer.com/article/10.1007/s10489-023-04574-9)." Applied Intelligence 53.18 (2023): 21116-21139.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/MOEADRL/An%20efficient%20evolutionary%20algorithm%20based%20on%20deep%20reinforcement%20learning%20for%20large-scale%20sparse%20multiobjective%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/MOEADRL/BibTex)|
|RLAM|Yin, Shiyuan, et al. "[**Reinforcement-learning-based parameter adaptation method for particle swarm optimization**](https://link.springer.com/article/10.1007/s40747-023-01012-8)." Complex & Intelligent Systems 9.5 (2023): 5585-5609.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLAM/Reinforcement-learning-based%20parameter%20adaptation%20method%20for%20particle%20swarm%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLAM/BibTex)|
|LDE|Sun, Jianyong, et al. "[**Learning Adaptive Differential Evolution Algorithm from Optimization Experiences by Policy Gradient**](https://ieeexplore.ieee.org/abstract/document/9359652)." IEEE Transactions on Evolutionary Computation 25.4 (2021): 666-680.|[yierh/LDE](https://github.com/yierh/LDE)|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/LDE/Learning%20Adaptive%20Differential%20Evolution%20Algorithm%20from%20Optimization%20Experiences%20by%20Policy%20Gradient.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/LDE/BibTex)|
|RL-PSO|Wu, Di, and G. Gary Wang. "[**Employing reinforcement learning to enhance particle swarm optimization methods**](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1867120)." Engineering Optimization 54.2 (2022): 329-348.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RL-PSO/Employing%20reinforcement%20learning%20to%20enhance%20particle%20swarm%20optimization%20methods.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RL-PSO/BibTex)|
|RLEPSO|Yin, Shiyuan, et al. "[**RLEPSO: Reinforcement learning based Ensemble particle swarm optimizer**](https://dl.acm.org/doi/abs/10.1145/3508546.3508599)." Proceedings of the 2021 4th International Conference on Algorithms, Computing and Artificial Intelligence. 2021.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLEPSO/RLEPSO%20Reinforcement%20learning%20based%20Ensemble%20particle%20swarm%20optimizer.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLEPSO/BibTex)|
|QLPSO|Xu, Yue, and Dechang Pi. "[**A reinforcement learning-based communication topology in particle swarm optimization**](https://link.springer.com/article/10.1007/s00521-019-04527-9)." Neural Computing and Applications 32 (2020): 10007-10032.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/QLPSO/A%20reinforcement%20learning-based%20communication%20topology%20in%20particle%20swarm%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/QLPSO/BibTex)|
|QLSOPSO \& QLMOPSO|Liu, Yaxian, et al. "[**An adaptive online parameter control algorithm for particle swarm optimization based on reinforcement learning**](https://ieeexplore.ieee.org/abstract/document/8790035/)." 2019 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2019.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/QLSOPSO\&QLMOPSO/An%20adaptive%20online%20parameter%20control%20algorithm%20for%20particle%20swarm%20optimization%20based%20on%20reinforcement%20learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/QLSOPSO\&QLMOPSO/BibTex)|
|RLMPSO|Samma, Hussein, Chee Peng Lim, and Junita Mohamad Saleh. "[**A new reinforcement learning-based memetic particle swarm optimizer**](https://www.sciencedirect.com/science/article/pii/S1568494616000132)." Applied Soft Computing 43 (2016): 276-297.|-|[PDF](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLMPSO/A%20new%20reinforcement%20learning-based%20memetic%20particle%20swarm%20optimizer.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Parameter%20Contorl/RLMPSO/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

#### 2.1.3. Operator \& Parameter

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|ALDes|Zhao, Qi, et al. "[**Automated Metaheuristic Algorithm Design with Autoregressive Learning**](https://arxiv.org/abs/2405.03419)." arXiv preprint arXiv:2405.03419 (2024).|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/ALDes/Automated%20Metaheuristic%20Algorithm%20Design%20with%20Autoregressive%20Learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/ALDes/BibTex)|
|MADAC|Xue, Ke, et al. "[**Multi-agent dynamic algorithm configuration**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022): 20147-20161.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/MADAC/Multi-agent%20dynamic%20algorithm%20configuration.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/MADAC/BibTex)|
|RL-HPSDE|Tan, Zhiping, et al. "[**Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning**](https://www.sciencedirect.com/science/article/pii/S2210650222001602)." Swarm and Evolutionary Computation 75 (2022): 101194.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/RL-HPSDE/Differential%20evolution%20with%20hybrid%20parameters%20and%20mutation%20strategies%20based%20on%20reinforcement%20learning.pdf)   [BibTex](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/RL-HPSDE/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

#### 2.1.4. Symbolic

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|SYMBOL|Chen, Jiacheng, et al. "[**Symbol: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning**](https://arxiv.org/abs/2402.02355)." The Twelfth International Conference on Learning Representations. 2024.|[GMC-DRL/Symbol](https://github.com/GMC-DRL/Symbol)|[PDF](MetaBBO/MetaBBO-RL/Symbolic/SYMBOL/SYMBOL%20Generating%20Flexible%20Black-Box%20Optimizers%20through%20Symbolic%20Equation%20Learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Symbolic/SYMBOL/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

#### 2.1.5. Others

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|UES-CMA-ES|Bolufé-Röhler, Antonio, and Bowen Xu. "[**Deep reinforcement learning for smart restarts in exploration-only exploitation-only metaheuristic hybrids**](https://link.springer.com/chapter/10.1007/978-3-031-62922-8_2)." Metaheuristics International Conference. Cham: Springer Nature Switzerland, 2024.|-|[PDF](MetaBBO/MetaBBO-RL/Others/UES-CMA-ES/Deep%20reinforcement%20learning%20for%20smart%20restarts%20in%20exploration-only%20exploitation-only%20metaheuristic%20hybrids.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/UES-CMA-ES/BibTex)|
|AGSEA|Shao, Shuai, Ye Tian, and Xingyi Zhang. "[**Deep reinforcement learning assisted automated guiding vector selection for large-scale sparse multi-objective optimization**](https://www.sciencedirect.com/science/article/pii/S2210650224001445)." Swarm and Evolutionary Computation 88 (2024): 101606.|-|[PDF](MetaBBO/MetaBBO-RL/Others/AGSEA/Deep%20Reinforcement%20Learning%20Assisted%20Automated%20Guiding%20Vector%20Selection%20for%20Large-scale%20Sparse%20Multi-objective%20Optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/AGSEA/BibTex)|
|MSORL|Wang, Xujie, et al. "[**A multi-swarm optimizer with a reinforcement learning mechanism for large-scale optimization**](https://www.sciencedirect.com/science/article/pii/S2210650224000191)." Swarm and Evolutionary Computation (2024): 101486.|-|[PDF](MetaBBO/MetaBBO-RL/Others/MSORL/A%20multi-swarm%20optimizer%20with%20a%20reinforcement%20learning%20mechanism%20for%20large-scale%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/MSORL/BibTex)|
|MELBA|Chaybouti, Sofian, et al. "[**Meta-learning of Black-box Solvers Using Deep Reinforcement Learning**](https://hal.science/hal-03930140/)." NeurIPS 2022, MetaLearn Workshop. 2022.|-|[PDF](MetaBBO/MetaBBO-RL/Others/MELBA/Meta-learning%20of%20Black-box%20Solvers%20Using%20Deep%20Reinforcement%20Learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/MELBA/BibTex)|
|LTO-POMDP|Gomes, Hugo Siqueira, Benjamin Léger, and Christian Gagné. "[**Meta learning black-box population-based optimizers**](https://arxiv.org/abs/2103.03526)." arXiv preprint arXiv:2103.03526 (2021).|[LTO-POMDP](https://github.com/optimization-toolbox/meta-learning-population-based-optimizers)|[PDF](MetaBBO/MetaBBO-RL/Others/LTO-POMDP/Meta%20learning%20black-box%20population-based%20optimizers.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/LTO-POMDP/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.2. MetaBBO with Supervised Learning (MetaBBO-SL)

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|GLHF|Li, Xiaobin, et al. "[**GLHF: General Learned Evolutionary Algorithm Via Hyper Functions**](https://arxiv.org/abs/2405.03728)." arXiv preprint arXiv:2405.03728 (2024).|-|[PDF](MetaBBO/MetaBBO-SL/GLHF/GLHF%20General%20Learned%20Evolutionary%20Algorithm%20Via%20Hyper%20Functions.pdf)  [BibTex](MetaBBO/MetaBBO-SL/GLHF/BibTex)|
|LEO|Yu, Peiyu, et al. "[**Latent Energy-Based Odyssey: Black-Box Optimization via Expanded Exploration in the Energy-Based Latent Space**](https://arxiv.org/abs/2405.16730)." arXiv preprint arXiv:2405.16730 (2024).|-|[PDF](MetaBBO/MetaBBO-SL/LEO/Latent%20Energy-Based%20Odyssey%20Black-Box%20Optimization%20via%20Expanded%20Exploration%20in%20the%20Energy-Based%20Latent%20Space.pdf)  [BibTex](MetaBBO/MetaBBO-SL/LEO/BibTex)|
|RIBBO|Song, Lei, et al. "[**Reinforced In-Context Black-Box Optimization**](https://arxiv.org/abs/2402.17423)." arXiv preprint arXiv:2402.17423 (2024).|[songlei00/RIBBO](https://github.com/songlei00/RIBBO)|[PDF](MetaBBO/MetaBBO-SL/RIBBO/Reinforced%20In-Context%20Black-Box%20Optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SL/RIBBO/BibTex)|
|NAP|Maraval, Alexandre, et al. "[**End-to-end meta-Bayesian optimisation with transformer neural processes**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2561721d0ca69bab22b749cfc4f48f6c-Abstract-Conference.html)." Advances in Neural Information Processing Systems 36 (2024).|-|[PDF](MetaBBO/MetaBBO-SL/NAP/End-to-end%20meta-Bayesian%20optimisation%20with%20transformer%20neural%20processes.pdf)  [BibTex](MetaBBO/MetaBBO-SL/NAP/BibTex)|
|OptFormer|Chen, Yutian, et al. "[**Towards learning universal hyperparameter optimizers with transformers**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf6501108fced72ee5c47e2151c4e153-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022): 32053-32068.|[google-research/optformer](https://github.com/google-research/optformer)|[PDF](MetaBBO/MetaBBO-SL/OptFormer/Towards%20learning%20universal%20hyperparameter%20optimizers%20with%20transformers.pdf)  [BibTex](MetaBBO/MetaBBO-SL/OptFormer/BibTex)|
|RNN-Opt|TV, Vishnu, et al. "[**Meta-learning for black-box optimization**](http://proceedings.mlr.press/v70/chen17e.html)." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer International Publishing, 2019.|-|[PDF](MetaBBO/MetaBBO-SL/RNN-Opt/Meta-learning%20for%20black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SL/RNN-Opt/BibTex)|
|RNN-OI|Chen, Yutian, et al. "[**Learning to learn without gradient descent by gradient descent**](http://proceedings.mlr.press/v70/chen17e.html)." International Conference on Machine Learning. PMLR, 2017.|-|[PDF](MetaBBO/MetaBBO-SL/RNN-OI/Learning%20to%20learn%20without%20gradient%20descent%20by%20gradient%20descent.pdf)  [BibTex](MetaBBO/MetaBBO-SL/RNN-OI/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.3. MetaBBO with Neuroevolution (MetaBBO-NE)

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|EvoTF|Lange, Robert Tjarko, Yingtao Tian, and Yujin Tang. "[**Evolution Transformer: In-Context Evolutionary Optimization**](https://arxiv.org/abs/2403.02985)." arXiv preprint arXiv:2403.02985 (2024).|[RobertTLange/evosax](https://github.com/RobertTLange/evosax)|[PDF](MetaBBO/MetaBBO-SR/EvoTF/Evolution%20Transformer%20In-Context%20Evolutionary%20Optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SR/EvoTF/BibTex)|
|LES|Lange, Robert, et al. "[**Discovering evolution strategies via meta-black-box optimization**](https://iclr.cc/virtual/2023/poster/11005)." The Eleventh International Conference on Learning Representations. 2023.|-|[PDF](MetaBBO/MetaBBO-SR/LES/Discovering%20evolution%20strategies%20via%20meta-black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SR/LES/BibTex)|
|LGA|Lange, Robert, et al. "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. 2023.|-|[PDF](MetaBBO/MetaBBO-SR/LGA/Discovering%20attention-based%20genetic%20algorithms%20via%20meta-black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SR/LGA/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.4. MetaBBO with LLMs

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|AS-LLM|Wu, Xingyu, et al. "[**Large language model-enhanced algorithm selection: towards comprehensive algorithm representation**](https://ira.lib.polyu.edu.hk/handle/10397/108348)." International Joint Conference on Artificial Intelligence, 2024.|-|[PDF](MetaBBO/MetaBBO-LLMs/AS-LLM/Large%20language%20model-enhanced%20algorithm%20selection%20towards%20comprehensive%20algorithm%20representation.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/AS-LLM/BibTex)|
|LLMOPT|Huang, Yuxiao, et al. "[**Towards Next Era of Multi-objective Optimization: Large Language Models as Architects of Evolutionary Operators**](https://arxiv.org/abs/2406.08987)." arXiv preprint arXiv:2406.08987 (2024).|-|[PDF](MetaBBO/MetaBBO-LLMs/LLMOPT/Towards%20Next%20Era%20of%20Multi-objective%20Optimization%20Large%20Language%20Models%20as%20Architects%20of%20Evolutionary%20Operators.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/LLMOPT/BibTex)|
|LLaMoCo|Ma, Zeyuan, et al. "[**LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation**](https://arxiv.org/abs/2403.01131)." arXiv preprint arXiv:2403.01131 (2024).|[LLaMoCo-722A](https://anonymous.4open.science/r/LLaMoCo-722A)|[PDF](MetaBBO/MetaBBO-LLMs/LLaMoCo/LLaMoCo%20Instruction%20Tuning%20of%20Large%20Language%20Models%20for%20Optimization%20Code%20Generation.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/LLaMoCo/BibTex)|
|LLaMEA|van Stein, Niki, and Thomas Bäck. "[**LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics**](https://arxiv.org/abs/2405.20132)." arXiv preprint arXiv:2405.20132 (2024).|-|[PDF](MetaBBO/MetaBBO-LLMs/LLaMEA/LLaMEA%20A%20Large%20Language%20Model%20Evolutionary%20Algorithm%20for%20Automatically%20Generating%20Metaheuristics.pdfLLaMoCo/LLaMoCo%20Instruction%20Tuning%20of%20Large%20Language%20Models%20for%20Optimization%20Code%20Generation.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/LLaMEA/BibTex)|
|EvoLLM|Lange, Robert Tjarko, Yingtao Tian, and Yujin Tang. "[**Large Language Models As Evolution Strategies**](https://arxiv.org/abs/2402.18381)." arXiv preprint arXiv:2402.18381 (2024).|-|[PDF](MetaBBO/MetaBBO-LLMs/EvoLLM/Large%20Language%20Models%20As%20Evolution%20Strategies.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/EvoLLM/BibTex)|
|CMOEA-LLM|Wang, Zeyi, et al. "[**Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization**](https://arxiv.org/abs/2405.05767)." arXiv preprint arXiv:2405.05767 (2024).|-|[PDF](MetaBBO/MetaBBO-LLMs/CMOEA-LLM/Large%20Language%20Model-Aided%20Evolutionary%20Search%20for%20Constrained%20Multiobjective%20Optimization.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/CMOEA-LLM/BibTex)|
|LEO|Brahmachary, Shuvayan, et al. "[**Large Language Model-Based Evolutionary Optimizer: Reasoning with elitism**](https://arxiv.org/abs/2403.02054)." arXiv preprint arXiv:2403.02054 (2024).|-|[PDF](MetaBBO/MetaBBO-LLMs/LEO/Large%20Language%20Model-Based%20Evolutionary%20Optimizer%20Reasoning%20with%20elitism.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/LEO/BibTex)|
|EvoPrompt|Guo, Qingyan, et al. "[**Connecting large language models with evolutionary algorithms yields powerful prompt optimizers**](https://openreview.net/forum?id=ZG3RaNIsO8)." The Twelfth International Conference on Learning Representations (2024).|[beeevita/EvoPrompt](https://github.com/beeevita/EvoPrompt)|[PDF](MetaBBO/MetaBBO-LLMs/EvoPrompt/Connecting%20large%20language%20models%20with%20evolutionary%20algorithms%20yields%20powerful%20prompt%20optimizers.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/EvoPrompt/BibTex)|
|Evoprompting|Chen, Angelica, David Dohan, and David So. "[**Evoprompting: Language models for code-level neural architecture search**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html)." Advances in Neural Information Processing Systems 36 (2024).|-|[PDF](MetaBBO/MetaBBO-LLMs/Evoprompting/EvoPrompting%20Language%20Models%20for%20Code-Level%20Neural%20Architecture%20Search.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/Evoprompting/BibTex)|
|Pluhacek, Michal, et al|Pluhacek, Michal, et al. "[**Leveraging large language models for the generation of novel metaheuristic optimization algorithms**](https://dl.acm.org/doi/abs/10.1145/3583133.3596401?casa_token=BbFUg5AKwPYAAAAA:AVnEru8jaHlfvs7tgYhiY8Qr4HBBvKTQxWA8xshnf2cdUSXfr6nlBQs6q0epy2iqw8-5m1RZ5VPW)." Proceedings of the Companion Conference on Genetic and Evolutionary Computation. 2023.|-|[PDF](MetaBBO/MetaBBO-LLMs/pluhacek2023leveraging/Leveraging%20Large%20Language%20Models%20for%20the%20Generation%20of%20Novel%20Metaheuristic%20Optimization%20Algorithms.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/pluhacek2023leveraging/BibTex)|
|LMEA|Liu, Shengcai, et al. "[**Large language models as evolutionary optimizers**](https://arxiv.org/abs/2310.19046)." arXiv preprint arXiv:2310.19046 (2023).|-|[PDF](MetaBBO/MetaBBO-LLMs/LMEA/Large%20Language%20Models%20as%20Evolutionary%20Optimizers.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/LMEA/BibTex)|
|AEL|Liu, Fei, et al. "[**Algorithm evolution using large language model**](https://arxiv.org/abs/2311.15249)." arXiv preprint arXiv:2311.15249 (2023).|-|[PDF](MetaBBO/MetaBBO-LLMs/AEL/Algorithm%20evolution%20using%20large%20language%20model.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/AEL/BibTex)|
|OPRO|Yang, Chengrun, et al. "[**Large language models as optimizers**](https://arxiv.org/abs/2309.03409)." arXiv preprint arXiv:2309.03409 (2023).|-|[PDF](MetaBBO/MetaBBO-LLMs/OPRO/Large%20Language%20Models%20as%20Optimizers.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/OPRO/BibTex)|
|Guo, Pei-Fu, et al|Guo, Pei-Fu, et al. "[**Towards optimizing with large language models**](https://arxiv.org/abs/2310.05204)." arXiv preprint arXiv:2310.05204 (2023).|-|[PDF](MetaBBO/MetaBBO-LLMs/guo2023towards/Towards%20Optimizing%20with%20Large%20Language%20Models.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/guo2023towards/BibTex)|
|OptiMUS|AhmadiTeshnizi, Ali, Wenzhi Gao, and Madeleine Udell. "[**OptiMUS: Optimization Modeling Using mip Solvers and large language models**](https://arxiv.org/abs/2310.06116)." arXiv preprint arXiv:2310.06116 (2023).|[teshnizi/OptiMUS](https://github.com/teshnizi/OptiMUS)|[PDF](MetaBBO/MetaBBO-LLMs/OptiMUS/OptiMUS%20Optimization%20Modeling%20Using%20MIP%20Solvers%20and%20large%20language%20models.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/OptiMUS/BibTex)|
|MOEA/D-LLM|Liu, Fei, et al. "[**Large language model for multi-objective evolutionary optimization**](https://arxiv.org/abs/2310.12541)." arXiv preprint arXiv:2310.12541 (2023).|-|[PDF](MetaBBO/MetaBBO-LLMs/MOEA-D-LLM/Large%20Language%20Model%20for%20Multi-objective%20Evolutionary%20Optimization.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/MOEA-D-LLM/BibTex)|
|EoH|Liu, Fei, et al. "[**Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model**](https://www.researchgate.net/publication/380399749_Evolution_of_Heuristics_Towards_Efficient_Automatic_Algorithm_Design_Using_Large_Language_Model)." arXiv preprint arXiv:2309.03409 (2023).|[nobodynobodypaper/EoH](https://github.com/nobodynobodypaper/EoH)|[PDF](MetaBBO/MetaBBO-LLMs/EoH/Evolution%20of%20Heuristics%20Towards%20Efficient%20Automatic%20Algorithm%20Design%20Using%20Large%20Language%20Model.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/EoH/BibTex)|
|Zhang, Michael R., et al|Zhang, Michael R., et al. "[**Using Large Language Models for Hyperparameter Optimization**](https://openreview.net/forum?id=FUdZ6HEOre)." NeurIPS 2023 Foundation Models for Decision Making Workshop. 2023.|-|[PDF](MetaBBO/MetaBBO-LLMs/zhang2023using/Using%20Large%20Language%20Models%20for%20Hyperparameter%20Optimization.pdf)  [BibTex](MetaBBO/MetaBBO-LLMs/zhang2023using/BibTex)|


See also [FeiLiu36/LLM4Opt](https://github.com/FeiLiu36/LLM4Opt) and [jxzhangjhu/Awesome-LLM-Prompt-Optimization](https://github.com/jxzhangjhu/Awesome-LLM-Prompt-Optimization).

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

## 2.5. Others
### 2.5.1 Evaluation Indicator
### 2.5.2 Landscape Feature


## 3. Classic BBO

### 3.1. Differential Evolution

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|ModDE|Vermetten, Diederick, et al. "[**Modular Differential Evolution**](https://arxiv.org/abs/2304.09524)." arXiv preprint arXiv:2304.09524 (2023).|[Dvermetten/ModDE](https://github.com/Dvermetten/ModDE)|[PDF](Classic%20BBO/Differential%20Evolution/ModDE/Modular%20Differential%20Evolution.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/ModDE/BibTex)|
|AMCDE|Ye, Chenxi, et al. "[**Differential evolution with alternation between steady monopoly and transient competition of mutation strategies**](https://www.sciencedirect.com/science/article/pii/S2210650223001761)." Swarm and Evolutionary Computation 83 (2023): 101403.|-|[PDF](Classic%20BBO/Differential%20Evolution/AMCDE/Differential%20evolution%20with%20alternation%20between%20steady%20monopoly%20and%20transient%20competition%20of%20mutation%20strategies.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/AMCDE/BibTex)|
|NL-SHADE-LBC|Stanovov, Vladimir, Akhmedova, Shakhnaz and Semenkin, Eugene "[**NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization**](https://ieeexplore.ieee.org/abstract/document/9870295/)." 2022 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2022.|-|[PDF](Classic%20BBO/Differential%20Evolution/NL-SHADE-LBC/NL-SHADE-LBC%20algorithm%20with%20linear%20parameter%20adaptation%20bias%20change%20for%20CEC%202022%20Numerical%20Optimization.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/NL-SHADE-LBC/BibTex)|
|MadDE|Biswas, Subhodip, et al. "[**Improving differential evolution through Bayesian hyperparameter optimization**](https://ieeexplore.ieee.org/abstract/document/9504792/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.|[subhodipbiswas/<br>MadDE](https://github.com/subhodipbiswas/MadDE)|[PDF](Classic%20BBO/Differential%20Evolution/MadDE/Improving%20differential%20evolution%20through%20Bayesian%20hyperparameter%20optimization.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/MadDE/BibTex)|
|jDE21|Brest, Janez, Mirjam Sepesy Maučec, and Borko Bošković. "[**Self-adaptive differential evolution algorithm with population size reduction for single objective bound-constrained optimization: Algorithm j21**](https://ieeexplore.ieee.org/abstract/document/9504782/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.|-|[PDF](Classic%20BBO/Differential%20Evolution/jDE21/Self-adaptive%20differential%20evolution%20algorithm%20with%20population%20size%20reduction%20for%20single%20objective%20bound-constrained%20optimization%20Algorithm%20j21.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/jDE21/BibTex)|
|NL-SHADE-RSP|Stanovov, Vladimir, Shakhnaz Akhmedova, and Eugene Semenkin. "[**NL-SHADE-RSP algorithm with adaptive archive and selective pressure for CEC 2021 numerical optimization**](https://ieeexplore.ieee.org/abstract/document/9504959/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.|-|[PDF](Classic%20BBO/Differential%20Evolution/NL-SHADE-RSP/NL-SHADE-RSP%20algorithm%20with%20adaptive%20archive%20and%20selective%20pressure%20for%20CEC%202021%20numerical%20optimization.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/NL-SHADE-RSP/BibTex)|
|EDEV|Wu, Guohua, et al. "[**Ensemble of differential evolution variants**](https://www.sciencedirect.com/science/article/pii/S0020025517309714)." Information Sciences 423 (2018): 172-186.|-|[PDF](Classic%20BBO/Differential%20Evolution/EDEV/Ensemble%20of%20differential%20evolution%20variants.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/EDEV/BibTex)|
|HMJCDE|Li, Genghui, et al. "[**A novel hybrid differential evolution algorithm with modified CoDE and JADE**](https://www.sciencedirect.com/science/article/pii/S1568494616302903)." Applied Soft Computing 47 (2016): 577-599.|-|[PDF](Classic%20BBO/Differential%20Evolution/HMJCDE/A%20novel%20hybrid%20differential%20evolution%20algorithm%20with%20modified%20CoDE%20and%20JADE.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/HMJCDE/BibTex)|
|L-SHADE|Tanabe, Ryoji, and Alex S. Fukunaga. "[**Improving the search performance of SHADE using linear population size reduction**](https://ieeexplore.ieee.org/abstract/document/6900380/)." 2014 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2014.|-|[PDF](Classic%20BBO/Differential%20Evolution/L-SHADE/Improving%20the%20search%20performance%20of%20SHADE%20using%20linear%20population%20size%20reduction.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/L-SHADE/BibTex)|
|SHADE|Tanabe, Ryoji, and Alex Fukunaga. "[**Success-history based parameter adaptation for differential evolution**](https://ieeexplore.ieee.org/abstract/document/6557555/)." 2013 IEEE Congress on Evolutionary Computation. IEEE, 2013.|-|[PDF](Classic%20BBO/Differential%20Evolution/SHADE/Success-history%20based%20parameter%20adaptation%20for%20differential%20evolution.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/SHADE/BibTex)|
|CoDE|Wang, Yong, Zixing Cai, and Qingfu Zhang. "[**Differential evolution with composite trial vector generation strategies and control parameters**](https://ieeexplore.ieee.org/abstract/document/5688232/)." IEEE Transactions on Evolutionary Computation 15.1 (2011): 55-66.|-|[PDF](Classic%20BBO/Differential%20Evolution/CoDE/Differential%20evolution%20with%20composite%20trial%20vector%20generation%20strategies%20and%20control%20parameters.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/CoDE/BibTex)|
|EPSDE|Mallipeddi, Rammohan, et al. "[**Differential evolution algorithm with ensemble of parameters and mutation strategies**](https://www.sciencedirect.com/science/article/pii/S1568494610001043)." Applied Soft Computing 11.2 (2011): 1679-1696.|-|[PDF](Classic%20BBO/Differential%20Evolution/EPSDE/Differential%20evolution%20algorithm%20with%20ensemble%20of%20parameters%20and%20mutation%20strategies.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/EPSDE/BibTex)|
|rJADE|Peng, Fei, et al. "[**Multi-start JADE with knowledge transfer for numerical optimization**](https://ieeexplore.ieee.org/abstract/document/4983171/)." 2009 IEEE Congress on Evolutionary Computation. IEEE, 2009.|-|[PDF](Classic%20BBO/Differential%20Evolution/rJADE/Multi-start%20JADE%20with%20knowledge%20transfer%20for%20numerical%20optimization.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/rJADE/BibTex)|
|JADE|Zhang, Jingqiao, and Arthur C. Sanderson. "[**JADE: adaptive differential evolution with optional external archive**](https://ieeexplore.ieee.org/abstract/document/5208221/)." IEEE Transactions on Evolutionary Computation 13.5 (2009): 945-958.|-|[PDF](Classic%20BBO/Differential%20Evolution/JADE/JADE%20adaptive%20differential%20evolution%20with%20optional%20external%20archive.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/JADE/BibTex)|
|jDE|Brest, Janez, et al. "[**Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems**](https://ieeexplore.ieee.org/abstract/document/4016057/)." IEEE Transactions on Evolutionary Computation 10.6 (2006): 646-657.|-|[PDF](Classic%20BBO/Differential%20Evolution/jDE/Self-adapting%20control%20parameters%20in%20differential%20evolution%20A%20comparative%20study%20on%20numerical%20benchmark%20problems.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/jDE/BibTex)|
|SaDE|Qin, A. Kai, and Ponnuthurai N. Suganthan. "[**Self-adaptive differential evolution algorithm for numerical optimization**](https://ieeexplore.ieee.org/abstract/document/1554904/)." 2005 IEEE Congress on Evolutionary Computation (CEC). Vol. 2. IEEE, 2005.|-|[PDF](Classic%20BBO/Differential%20Evolution/SaDE/Self-adaptive%20differential%20evolution%20algorithm%20for%20numerical%20optimization.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/SaDE/BibTex)|
|Vanilla DE|Storn, Rainer, and Kenneth Price. "[**Differential evolution-a simple and efficient heuristic for global optimization over continuous spaces**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of Global Optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/Differential%20evolution-a%20simple%20and%20efficient%20heuristic%20for%20global%20optimization%20over%20continuous%20spaces.pdf)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 3.2. Partical Swarm Optimization

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|SAHLPSO|Tao, Xinmin, et al. "[**Self-Adaptive two roles hybrid learning strategies-based particle swarm optimization**](https://www.sciencedirect.com/science/article/pii/S0020025521006988)." Information Sciences 578 (2021): 457-481.|-|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/SAHLPSO/Self-Adaptive%20two%20roles%20hybrid%20learning%20strategies-based%20particle%20swarm%20optimization.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/SAHLPSO/BibTex)|
|EPSO|Lynn, Nandar, and Ponnuthurai Nagaratnam Suganthan. "[**Ensemble particle swarm optimizer**](https://www.sciencedirect.com/science/article/pii/S1568494617300753)." Applied Soft Computing 55 (2017): 533-548.|-|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/EPSO/Ensemble%20particle%20swarm%20optimizer.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/EPSO/BibTex)|
|GLPSO|Yue-Jiao Gong, Jing-Jing Li, Yicong Zhou, Yun Li, Henry Shu-Hung Chung, Yu-hui Shi, Jun Zhang. "[**Genetic learning particle swarm optimization**](https://ieeexplore.ieee.org/abstract/document/7271066/)." IEEE Transactions on Cybernetics 46.10 (2015): 2277-2290.|[YuejiaoGong/<br>genetic_learning_PSO](https://github.com/YuejiaoGong/genetic_learning_PSO)|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/GLPSO/Genetic%20learning%20particle%20swarm%20optimization.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/GLPSO/BibTex)|
|sDMS-PSO|Liang, Jing J., et al. "[**A self-adaptive dynamic particle swarm optimizer**](https://ieeexplore.ieee.org/abstract/document/7257290/)." 2015 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2015.|-|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/sDMS-PSO/A%20self-adaptive%20dynamic%20particle%20swarm%20optimizer.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/sDMS-PSO/BibTex)|
|DMS-PSO|Liang, Jane-Jing, and Ponnuthurai Nagaratnam Suganthan. "[**Dynamic multi-swarm particle swarm optimizer**](https://ieeexplore.ieee.org/abstract/document/1501611/)." Proceedings 2005 IEEE Swarm Intelligence Symposium, 2005. SIS 2005.. IEEE, 2005.|-|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/DMS-PSO/Dynamic%20multi-swarm%20particle%20swarm%20optimizer.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/DMS-PSO/BibTex)|
|FIPSO|Mendes, Rui, James Kennedy, and José Neves. "[**The fully informed particle swarm: simpler, maybe better**](https://ieeexplore.ieee.org/abstract/document/1304843/)." IEEE Transactions on Evolutionary Computation 8.3 (2004): 204-210.|-|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/FIPSO/The%20fully%20informed%20particle%20swarm%20simpler%2C%20maybe%20better.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/FIPSO/BibTex)|
|Vanilla PSO|Kennedy, James, and Russell Eberhart. "[**Particle swarm optimization**](https://ieeexplore.ieee.org/abstract/document/488968/)." Proceedings of ICNN'95-International Conference on Neural Networks. Vol. 4. IEEE, 1995.|-|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/Vanilla%20PSO/Particle%20swarm%20optimization.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/Vanilla%20PSO/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 3.3. Evolution Strategies

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|PSA-CMA-ES|Nishida, Kouhei, and Youhei Akimoto. "[**Psa-cma-es: Cma-es with population size adaptation**](https://dl.acm.org/doi/abs/10.1145/3205455.3205467)." Proceedings of the Genetic and Evolutionary Computation Conference. 2018.|-|[PDF](Classic%20BBO/Evolution%20Strategies/PSA-CMA-ES/Psa-cma-es%20Cma-es%20with%20population%20size%20adaptation.pdf)  [BibTex](Classic%20BBO/Evolution%20Strategies/PSA-CMA-ES/BibTex)|
|CC-CMA-ES|Liu, Jinpeng, and Ke Tang. "[**Scaling up covariance matrix adaptation evolution strategy using cooperative coevolution**](https://link.springer.com/chapter/10.1007/978-3-642-41278-3_43)." International Conference on Intelligent Data Engineering and Automated Learning. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013.|-|[PDF](Classic%20BBO/Evolution%20Strategies/CC-CMA-ES/Scaling%20up%20covariance%20matrix%20adaptation%20evolution%20strategy%20using%20cooperative%20coevolution.pdf)  [BibTex](Classic%20BBO/Evolution%20Strategies/CC-CMA-ES/BibTex)|
|BIPOP-CMA-ES|Hansen, Nikolaus. "[**Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed**](https://dl.acm.org/doi/abs/10.1145/1570256.1570333)." Proceedings of the 11th Annual Conference Companion on Genetic and Evolutionary Computation Conference: late breaking papers. 2009.|-|[PDF](Classic%20BBO/Evolution%20Strategies/BIPOP-CMA-ES/Benchmarking%20a%20BI-population%20CMA-ES%20on%20the%20BBOB-2009%20function%20testbed.pdf)  [BibTex](Classic%20BBO/Evolution%20Strategies/BIPOP-CMA-ES/BibTex)|
|IPOP-CMA-ES|Auger, Anne, and Nikolaus Hansen. "[**"A restart CMA evolution strategy with increasing population size**](https://ieeexplore.ieee.org/abstract/document/1554902/)." 2005 IEEE Congress on Evolutionary Computation (CEC). Vol. 2. IEEE, 2005.|-|[PDF](Classic%20BBO/Evolution%20Strategies/IPOP-CMA-ES/A%20restart%20CMA%20evolution%20strategy%20with%20increasing%20population%20size.pdf)  [BibTex](Classic%20BBO/Evolution%20Strategies/IPOP-CMA-ES/BibTex)|
|CMA-ES|Hansen, Nikolaus, Sibylle D. Müller, and Petros Koumoutsakos. "[**Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)**](https://ieeexplore.ieee.org/abstract/document/6790790/)." Evolutionary Computation 11.1 (2003): 1-18.|-|[PDF](Classic%20BBO/Evolution%20Strategies/CMA-ES/Reducing%20the%20time%20complexity%20of%20the%20derandomized%20evolution%20strategy%20with%20covariance%20matrix%20adaptation%20(CMA-ES).pdf)  [BibTex](Classic%20BBO/Evolution%20Strategies/CMA-ES/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 3.4. Bayesian Optimization

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|BO|Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. "[**Practical bayesian optimization of machine learning algorithms**](https://proceedings.neurips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html)." Advances in Neural Information Processing Systems 25 (2012).|-|[PDF](Classic%20BBO/Bayesian%20Optimization/BO/Practical%20bayesian%20optimization%20of%20machine%20learning%20algorithms.pdf)  [BibTex](Classic%20BBO/Bayesian%20Optimization/BO/BibTex)|
|SMAC3|Lindauer, Marius, et al. "[**SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization**](https://jmlr.org/papers/v23/21-0888.html)." The Journal of Machine Learning Research 23.1 (2022): 2475-2483.|[automl/SMAC3](https://github.com/automl/SMAC3)|[PDF](Classic%20BBO/Bayesian%20Optimization/SMAC/SMAC3%20A%20Versatile%20Bayesian%20Optimization%20Package%20for%20Hyperparameter%20Optimization.pdf)  [BibTex](Classic%20BBO/Bayesian%20Optimization/SMAC/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 3.5. Others

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|MFEA(-II)|Gupta, Abhishek, Yew-Soon Ong, and Liang Feng. "[**Multifactorial evolution: toward evolutionary multitasking**](https://ieeexplore.ieee.org/abstract/document/7161358/)." IEEE Transactions on Evolutionary Computation 20.3 (2015): 343-357.<br>Bali, Kavitesh Kumar, et al. "[**Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II**](https://ieeexplore.ieee.org/abstract/document/8672822/)." IEEE Transactions on Evolutionary Computation 24.1 (2019): 69-83.|-|[PDF](Classic%20BBO/Others/MFEA/)  [BibTex](Classic%20BBO/Others/MFEA/BibTex)|
|MOEA/D|Zhang, Qingfu, and Hui Li. "[**MOEA/D: A multiobjective evolutionary algorithm based on decomposition**](https://ieeexplore.ieee.org/abstract/document/4358754/)." IEEE Transactions on Evolutionary Computation 11.6 (2007): 712-731.|-|[PDF](Classic%20BBO/Others/MOEA-D/MOEA-D%20A%20multiobjective%20evolutionary%20algorithm%20based%20on%20decomposition.pdf)  [BibTex](Classic%20BBO/Others/MOEA-D/BibTex)|
|VNCDE|Zhang, Yu-Hui, et al. "[**Parameter-free voronoi neighborhood for evolutionary multimodal optimization**](https://ieeexplore.ieee.org/abstract/document/8738874/)." IEEE Transactions on Evolutionary Computation 24.2 (2019): 335-349.|-|[PDF](Classic%20BBO/Others/VNCDE/Parameter-free%20voronoi%20neighborhood%20for%20evolutionary%20multimodal%20optimization.pdf)  [BibTex](Classic%20BBO/Others/VNCDE/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

