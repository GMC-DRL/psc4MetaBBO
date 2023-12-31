# Useful Papers and Source Codes for Meta Black-Box Optimization (MetaBBO)

This respository aims to maintain a list of useful relevant papers and open source codes for MetaBBO. Our implementations of some of these methods can be accessed in [MetaBox](https://github.com/GMC-DRL/MetaBox).

- [1. Survey Papers \& Benchmarks](#1-survey-papers--benchmarks)
  - [1.1. Survey Papers](#11-survey-papers)
  - [1.2. Benchmarks](#12-benchmarks)
- [2. MetaBBO](#2-metabbo)
  - [2.1. MetaBBO with Reinforcement Learning](#21-metabbo-with-reinforcement-learning-metabbo-rl)
    - [2.1.1. Operator Selection](#211-operator-selection)
    - [2.1.2. Parameter Contorl](#212-parameter-contorl)
    - [2.1.3. Operator \& Parameter](#213-operator--parameter)
    - [2.1.4. Others](#214-others)
  - [2.2. MetaBBO with Supervised Learning](#22-metabbo-with-supervised-learning-metabbo-sl)
  - [2.3. MetaBBO with Self-Referential Search](#23-metabbo-with-self-referential-search-metabbo-sr)
  - [2.4. Other MetaBBO](#24-other-metabbo)
- [3. Classic BBO](#3-classic-bbo)
  - [3.1. Differential Evolution](#31-differential-evolution)
  - [3.2. Partical Swarm Optimization](#32-partical-swarm-optimization)
  - [3.3. Evolution Strategies](#33-evolution-strategies)
  - [3.4. Bayesian Optimization](#34-bayesian-optimization)
  - [3.5. Others](#35-others)


## 1. Survey Papers \& Benchmarks

### 1.1. Survey Papers

|Paper|About|
|:-:|:-:|
|Chernigovskaya, Maria, Andrey Kharitonov, and Klaus Turowski. "[**A Recent Publications Survey on Reinforcement Learning for Selecting Parameters of Meta-Heuristic and Machine Learning Algorithms**](https://www.scitepress.org/Papers/2023/119543/119543.pdf)." CLOSER. 2023.|[PDF](Survey/Chernigovskaya%20et%20al/A%20Recent%20Publications%20Survey%20on%20Reinforcement%20Learning%20for%20Selecting%20Parameters%20of%20Meta-Heuristic%20and%20Machine%20Learning%20Algorithms.pdf)  [BibTex](Survey/Chernigovskaya%20et%20al/BibTex)|
|Drugan, Madalina M. "[**Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms**](https://www.sciencedirect.com/science/article/abs/pii/S2210650217302766)." Swarm and Evolutionary Computation 44 (2019): 228-246.|[PDF](Survey/Drugan%20et%20al/Reinforcement%20learning%20versus%20evolutionary%20computation%20A%20survey%20on%20hybrid%20algorithms.pdf)  [BibTex](Survey/Drugan%20et%20al/BibTex)|

### 1.2. Benchmarks

|Benchmark|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|MetaBox|Ma, Zeyuan, et al. "[**MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning**](https://neurips.cc/virtual/2023/oral/73737)." Advances in Neural Information Processing Systems 36 (2023).|[GMC-DRL/MetaBox]( https://github.com/GMC-DRL/MetaBox)|[PDF](Benchmarks/MetaBox/MetaBox%20A%20Benchmark%20Platform%20for%20Meta-Black-Box%20Optimization%20with%20Reinforcement%20Learning.pdf)  [BibTex](Benchmarks/MetaBox/BibTex)|
|COCO|Hansen, Nikolaus, et al. "[**COCO: A platform for comparing continuous optimizers in a black-box setting**](https://www.tandfonline.com/doi/abs/10.1080/10556788.2020.1808977)." Optimization Methods and Software 36.1 (2021): 114-144.|[numbbo/coco](https://github.com/numbbo/coco)|[PDF](Benchmarks/COCO/COCO%20A%20platform%20for%20comparing%20continuous%20optimizers%20in%20a%20black-box%20setting.pdf)  [BibTex](Benchmarks/COCO/BibTex)|
|IOHprofiler (IOHexperimenter)|Doerr, Carola, et al. "[**IOHprofiler: A benchmarking and profiling tool for iterative optimization heuristics**](https://arxiv.org/abs/1810.05281)." arXiv preprint arXiv:1810.05281 (2018).<br>de Nobel, Jacob, et al. "[**Iohexperimenter: Benchmarking platform for iterative optimization heuristics**](https://direct.mit.edu/evco/article/doi/10.1162/evco_a_00342/116949)." Evolutionary Computation (2023): 1-6.|[IOHprofiler/<br>IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter)|[PDF](Benchmarks/IOHprofiler/)  [BibTex](Benchmarks/IOHprofiler/BibTex)|
|AClib|Hutter, Frank, et al. "[**AClib: A benchmark library for algorithm configuration**](https://link.springer.com/chapter/10.1007/978-3-319-09584-4_4)." Learning and Intelligent Optimization: 8th International Conference. Revised Selected Papers 8. Springer International Publishing, 2014.|[aclib.net](https://www.aclib.net/)|[PDF](Benchmarks/AClib/AClib%20A%20benchmark%20library%20for%20algorithm%20configuration.pdf)  [BibTex](Benchmarks/AClib/BibTex)|
|Olympus|Häse, Florian, et al. "[**Olympus: a benchmarking framework for noisy optimization and experiment planning**](https://iopscience.iop.org/article/10.1088/2632-2153/abedc8/meta)." Machine Learning: Science and Technology 2.3 (2021): 035021.|[aspuru-guzik-group/olympus](https://github.com/aspuru-guzik-group/olympus)|[PDF](Benchmarks/Olympus/Olympus%20a%20benchmarking%20framework%20for%20noisy%20optimization%20and%20experiment%20planning.pdf)  [BibTex](Benchmarks/Olympus/BibTex)|
|Bayesmark|Turner, R., and D. Eriksson. "**Bayesmark: Benchmark framework to easily compare bayesian optimization methods on real machine learning tasks**." (2019).|[uber/bayesmark](https://github.com/uber/bayesmark)|[Doc](https://bayesmark.readthedocs.io/en/latest/)  [BibTex](Benchmarks/Bayesmark/BibTex)|
|IEEE CEC 2022|Abhishek Kumar, Kenneth V. Price, Ali Wagdy Mohamed, Anas A. Hadi, P. N. Suganthan, "[**Problem definitions and evaluation criteria for the cec 2022 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2022/CEC2022.htm)." Technical Report, Nanyang Technological University, Singapore, 2022|[P-N-Suganthan/2022-SO-BO](https://github.com/P-N-Suganthan/2022-SO-BO)|[PDF](Benchmarks/CEC/2022/Problem%20definitions%20and%20evaluation%20criteria%20for%20the%20cec%202022%20Special%20Session%20and%20Competition%20on%20Single%20Objective%20Bound%20Constrained%20Numerical%20Optimization.pdf)  [BibTex](Benchmarks/CEC/2022/BibTex)|
|IEEE CEC 2021|Ali Wagdy, Anas A Hadi, Ali K. Mohamed, Prachi Agrawal, Abhishek Kumar and P. N. Suganthan, "[**Problem definitions and evaluation criteria for the cec 2021 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm)." Technical Report, Nanyang Technological University, Singapore, 2021|[P-N-Suganthan/2021-SO-BCO](https://github.com/P-N-Suganthan/2021-SO-BCO)|[PDF](Benchmarks/CEC/2021/Problem%20definitions%20and%20evaluation%20criteria%20for%20the%20cec%202021%20Special%20Session%20and%20Competition%20on%20Single%20Objective%20Bound%20Constrained%20Numerical%20Optimization.pdf)  [BibTex](Benchmarks/CEC/2021/BibTex)|
|IEEE CEC 2017|N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu and P. N. Suganthan, "[**Problem definitions and evaluation criteria for the CEC 2017 competition on constrained real-parameter optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)." Technical Report, Nanyang Technological University, Singapore, 2017|[P-N-Suganthan/CEC2017-BoundContrained](https://github.com/P-N-Suganthan/CEC2017-BoundContrained)|[PDF](Benchmarks/CEC/2017/Problem%20definitions%20and%20evaluation%20criteria%20for%20the%20CEC%202017%20competition%20on%20constrained%20real-parameter%20optimization.pdf)  [BibTex](Benchmarks/CEC/2017/BibTex)|
|IEEE CEC 2015|J. J. Liang, B. Y. Qu, P. N. Suganthan, Q. Chen, "[**Problem Definitions and Evaluation Criteria for the CEC 2015 Competition on Learning-based Real-Parameter Single Objective Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2015/CEC2015.htm)", Technical Report, Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou, China and  Technical Report, Nanyang Technological University, Singapore, 2015.|[P-N-Suganthan/CEC2015-Learning-Based](https://github.com/P-N-Suganthan/CEC2015-Learning-Based)|[PDF](Benchmarks/CEC/2015/Problem%20Definitions%20and%20Evaluation%20Criteria%20for%20the%20CEC%202015%20Competition%20on%20Learning-based%20Real-Parameter%20Single%20Objective%20Optimization.pdf)  [BibTex](Benchmarks/CEC/2015/BibTex)|
|IEEE CEC 2013|J. J. Liang, B-Y. Qu, P. N. Suganthan, Alfredo G. Hernández-Díaz, "[**Problem Definitions and Evaluation Criteria for the CEC 2013 Special Session and Competition on Real-Parameter Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2013/CEC2013.htm)", Technical Report, Computational Intelligence Laboratory, Zhengzhou University, Zhengzhou China and  Technical Report, Nanyang Technological University, Singapore, 2013.|[P-N-Suganthan/CEC2013](https://github.com/P-N-Suganthan/CEC2013)|[PDF](Benchmarks/CEC/2013/Problem%20Definitions%20and%20Evaluation%20Criteria%20for%20the%20CEC%202013%20Special%20Session%20and%20Competition%20on%20Real-Parameter%20Optimization.pdf)  [BibTex](Benchmarks/CEC/2013/BibTex)|
|Zigzag BBO|Kudela, Jakub. "[**Novel zigzag-based benchmark functions for bound constrained single objective optimization**](https://ieeexplore.ieee.org/abstract/document/9504720/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.<br>Kudela, Jakub, and Radomil Matousek. "[**New benchmark functions for single-objective optimization based on a zigzag pattern**](https://ieeexplore.ieee.org/abstract/document/9684455/)." IEEE Access 10 (2022): 8262-8278.|[JakubKudela89/Zigzag](https://github.com/JakubKudela89/Zigzag)|[PDF](Benchmarks/Zigzag/)  [BibTex](Benchmarks/Zigzag/BibTex)|
|Protein–Docking|Hwang, Howook, et al. "[**Protein–protein docking benchmark version 4.0**](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22830)." Proteins: Structure, Function, and Bioinformatics 78.15 (2010): 3111-3114.|[Protein–Docking](http://zlab.umassmed.edu/benchmark/)|[PDF](Benchmarks/Protein–Docking/Protein--protein%20docking%20benchmark%20version%204.0.pdf)  [BibTex](Benchmarks/Protein–Docking/BibTex)|
|HPOBench|Eggensperger, Katharina, et al. "[**HPOBench: A collection of reproducible multi-fidelity benchmark problems for HPO**](https://arxiv.org/abs/2109.06716)." arXiv preprint arXiv:2109.06716 (2021).|[automl/HPOBench](https://github.com/automl/HPOBench)|[PDF](Benchmarks/HPOBench/HPOBench%20A%20collection%20of%20reproducible%20multi-fidelity%20benchmark%20problems%20for%20HPO.pdf)  [BibTex](Benchmarks/HPOBench/BibTex)|
|DACBench|Eimer, Theresa, et al. "[**DACBench: A benchmark library for dynamic algorithm configuration**](https://arxiv.org/abs/2105.08541)." arXiv preprint arXiv:2105.08541 (2021).|[automl/DACBench](https://github.com/automl/DACBench)|[PDF](Benchmarks/DACBench/DACBench%20A%20benchmark%20library%20for%20dynamic%20algorithm%20configuration.pdf)  [BibTex](Benchmarks/DACBench/BibTex)|

**The complete list of IEEE CEC series can be access at [ntu.edu.sg](https://www3.ntu.edu.sg/home/epnsugan/index_files/).*

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>


## 2. MetaBBO

### 2.1. MetaBBO with Reinforcement Learning (MetaBBO-RL)

#### 2.1.1. Operator Selection

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|RLDMDE|Yang, Qingyong, et al. "[**Dynamic multi-strategy integrated differential evolution algorithm based on reinforcement learning for optimization problems**](https://link.springer.com/article/10.1007/s40747-023-01243-9)." Complex & Intelligent Systems (2023): 1-33.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20Selection/RLDMDE/Dynamic%20multi-strategy%20integrated%20differential%20evolution%20algorithm%20based%20on%20reinforcement%20learning%20for%20optimization%20problems.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20Selection/RLDMDE/BibTex)|
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
|MADAC|Xue, Ke, et al. "[**Multi-agent dynamic algorithm configuration**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022): 20147-20161.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/MADAC/Multi-agent%20dynamic%20algorithm%20configuration.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/MADAC/BibTex)|
|RL-HPSDE|Tan, Zhiping, et al. "[**Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning**](https://www.sciencedirect.com/science/article/pii/S2210650222001602)." Swarm and Evolutionary Computation 75 (2022): 101194.|-|[PDF](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/RL-HPSDE/Differential%20evolution%20with%20hybrid%20parameters%20and%20mutation%20strategies%20based%20on%20reinforcement%20learning.pdf)   [BibTex](MetaBBO/MetaBBO-RL/Operator%20%26%20Parameter/RL-HPSDE/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

#### 2.1.4. Others

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|MELBA|Chaybouti, Sofian, et al. "[**Meta-learning of Black-box Solvers Using Deep Reinforcement Learning**](https://hal.science/hal-03930140/)." NeurIPS 2022, MetaLearn Workshop. 2022.|-|[PDF](MetaBBO/MetaBBO-RL/Others/MELBA/Meta-learning%20of%20Black-box%20Solvers%20Using%20Deep%20Reinforcement%20Learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/MELBA/BibTex)|
|LTO-POMDP|Gomes, Hugo Siqueira, Benjamin Léger, and Christian Gagné. "[**Meta learning black-box population-based optimizers**](https://arxiv.org/abs/2103.03526)." arXiv preprint arXiv:2103.03526 (2021).|[LTO-POMDP](https://github.com/optimization-toolbox/meta-learning-population-based-optimizers)|[PDF](MetaBBO/MetaBBO-RL/Others/LTO-POMDP/Meta%20learning%20black-box%20population-based%20optimizers.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/LTO-POMDP/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.2. MetaBBO with Supervised Learning (MetaBBO-SL)

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|RNN-Opt|TV, Vishnu, et al. "[**Meta-learning for black-box optimization**](http://proceedings.mlr.press/v70/chen17e.html)." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer International Publishing, 2019.|-|[PDF](MetaBBO/MetaBBO-SL/RNN-Opt/Meta-learning%20for%20black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SL/RNN-Opt/BibTex)|
|RNN-OI|Chen, Yutian, et al. "[**Learning to learn without gradient descent by gradient descent**](http://proceedings.mlr.press/v70/chen17e.html)." International Conference on Machine Learning. PMLR, 2017.|-|[PDF](MetaBBO/MetaBBO-SL/RNN-OI/Learning%20to%20learn%20without%20gradient%20descent%20by%20gradient%20descent.pdf)  [BibTex](MetaBBO/MetaBBO-SL/RNN-OI/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.3. MetaBBO with Self-Referential Search (MetaBBO-SR)

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|LES|Lange, Robert, et al. "[**Discovering evolution strategies via meta-black-box optimization**](https://iclr.cc/virtual/2023/poster/11005)." The Eleventh International Conference on Learning Representations. 2023.|-|[PDF](MetaBBO/MetaBBO-SR/LES/Discovering%20evolution%20strategies%20via%20meta-black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SR/LES/BibTex)|
|LGA|Lange, Robert, et al. "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. 2023.|-|[PDF](MetaBBO/MetaBBO-SR/LGA/Discovering%20attention-based%20genetic%20algorithms%20via%20meta-black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SR/LGA/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.4. Other MetaBBO

## 3. Classic BBO

### 3.1. Differential Evolution

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
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
|Vanilla PSO|Kennedy, James, and Russell Eberhart. "[**Particle swarm optimization**](https://ieeexplore.ieee.org/abstract/document/488968/)." Proceedings of ICNN'95-International Conference on Neural Networks. Vol. 4. IEEE, 1995.|-|[PDF](Classic%20BBO/Partical%20Swarm%20Optimization/Vanilla%20PSO/Particle%20swarm%20optimization.pdf)  [BibTex](Classic%20BBO/Partical%20Swarm%20Optimization/RLDMDE/BibTex)|

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
|SMAC3|Lindauer, Marius, et al. "[**SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization**](https://jmlr.org/papers/v23/21-0888.html)." The Journal of Machine Learning Research 23.1 (2022): 2475-2483.|[automl/SMAC3](https://github.com/automl/SMAC3)|[PDF](Classic%20BBO/Bayesian%20Optimization/SMAC3/SMAC3%20A%20Versatile%20Bayesian%20Optimization%20Package%20for%20Hyperparameter%20Optimization.pdf)  [BibTex](Classic%20BBO/Bayesian%20Optimization/SMAC3/BibTex)|

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

