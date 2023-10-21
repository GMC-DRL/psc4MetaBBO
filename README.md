# Useful Papers and Source Codes for Meta Black-Box Optimization (MetaBBO)

This respository aims to maintain a list of useful relevant papers and open source codes for MetaBBO. Our implementations of these methods can be accessed in [MetaBox](https://github.com/GMC-DRL/MetaBox).

- [1. Survey Papers](#1-survey-papers)
- [2. Classic BBO](#2-classic-bbo)
  - [2.1. Differential Evolution](#21-differential-evolution)
  - [2.2. Partical Swarm Optimization](#22-partical-swarm-optimization)
  - [2.3. Evolution Strategies](#23-evolution-strategies)
  - [2.4. Bayesian Optimization](#24-bayesian-optimization)
  - [2.5. Others](#25-others)
- [3. MetaBBO](#3-metabbo)
  - [3.1. MetaBBO with Reinforcement Learning](#31-metabbo-with-reinforcement-learning-metabbo-rl)
    - [3.1.1. Differential Evolution](#311-differential-evolution)
    - [3.1.2. Partical Swarm Optimization](#312-partical-swarm-optimization)
    - [3.1.3. Others](#313-others)
  - [3.2. MetaBBO with Supervised Learning](#32-metabbo-with-supervised-learning-metabbo-sl)
  - [3.3. MetaBBO with Self-Referential Search](#33-metabbo-with-self-referential-search-metabbo-sr)
  - [3.4. Other MetaBBO](#34-other-metabbo)
- [4. Benchmarks](#4-benchmarks)

## 1. Survey Papers

|Paper|About|
|:-:|:-:|
|Chernigovskaya, Maria, Andrey Kharitonov, and Klaus Turowski. "[**A Recent Publications Survey on Reinforcement Learning for Selecting Parameters of Meta-Heuristic and Machine Learning Algorithms**](https://www.scitepress.org/Papers/2023/119543/119543.pdf)." CLOSER. 2023.|[PDF](Survey/Chernigovskaya%20et%20al/A%20Recent%20Publications%20Survey%20on%20Reinforcement%20Learning%20for%20Selecting%20Parameters%20of%20Meta-Heuristic%20and%20Machine%20Learning%20Algorithms.pdf)  [BibTex](Survey/Chernigovskaya%20et%20al/BibTex)|
|Drugan, Madalina M. "[**Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms**](https://www.sciencedirect.com/science/article/abs/pii/S2210650217302766)." Swarm and Evolutionary Computation 44 (2019): 228-246.|[PDF](Survey/Drugan%20et%20al/Reinforcement%20learning%20versus%20evolutionary%20computation%20A%20survey%20on%20hybrid%20algorithms.pdf)  [BibTex](Survey/Drugan%20et%20al/BibTex)|

## 2. Classic BBO

### 2.1. Differential Evolution

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|Vanilla DE|Storn, Rainer, and Kenneth Price "[**Differential evolution-a simple and efficient heuristic for global optimization over continuous spaces**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|jDE|Storn, Rainer, and Kenneth Price "[**Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|JADE|Storn, Rainer, and Kenneth Price "[**JADE: adaptive differential evolution with optional external archive**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|rJADE|Storn, Rainer, and Kenneth Price "[**Multi-start JADE with knowledge transfer for numerical optimization**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|jDE21|Storn, Rainer, and Kenneth Price "[**Self-adaptive differential evolution algorithm with population size reduction for single objective bound-constrained optimization: Algorithm j21**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|SHADE|Storn, Rainer, and Kenneth Price "[**Success-history based parameter adaptation for differential evolution**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|L-SHADE|Storn, Rainer, and Kenneth Price "[**Improving the search performance of SHADE using linear population size reduction**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|NL-SHADE-RSP|Storn, Rainer, and Kenneth Price "[**NL-SHADE-RSP algorithm with adaptive archive and selective pressure for CEC 2021 numerical optimization**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|NL-SHADE-LBC|Storn, Rainer, and Kenneth Price "[**NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|MadDE|Storn, Rainer, and Kenneth Price "[**Improving differential evolution through Bayesian hyperparameter optimization**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|SaDE|Storn, Rainer, and Kenneth Price "[**Self-adaptive differential evolution algorithm for numerical optimization**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|
|CoDE|Storn, Rainer, and Kenneth Price "[**Differential evolution with composite trial vector generation strategies and control parameters**](https://link.springer.com/article/10.1023/a:1008202821328)." Journal of global optimization 11.4 (1997): 341.|-|[PDF](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/)  [BibTex](Classic%20BBO/Differential%20Evolution/Vanilla%20DE/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 2.2. Partical Swarm Optimization

### 2.3. Evolution Strategies

### 2.4. Bayesian Optimization

### 2.5. Others

## 3. MetaBBO

### 3.1. MetaBBO with Reinforcement Learning (MetaBBO-RL)

#### 3.1.1. Differential Evolution

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|*RLDMDE|Yang, Qingyong, et al. "[**Dynamic multi-strategy integrated differential evolution algorithm based on reinforcement learning for optimization problems**](https://link.springer.com/article/10.1007/s40747-023-01243-9)." Complex & Intelligent Systems (2023): 1-33.|-|[PDF](MetaBBO/MetaBBO-RL/Differential%20Evolution/RLDMDE/Dynamic%20multi-strategy%20integrated%20differential%20evolution%20algorithm%20based%20on%20reinforcement%20learning%20for%20optimization%20problems.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Differential%20Evolution/RLDMDE/BibTex)|
|RL-HPSDE|Tan, Zhiping, et al. "[**Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning**](https://www.sciencedirect.com/science/article/pii/S2210650222001602)." Swarm and Evolutionary Computation 75 (2022): 101194.|-|[PDF](MetaBBO/MetaBBO-RL/Differential%20Evolution/RL-HPSDE/Differential%20evolution%20with%20hybrid%20parameters%20and%20mutation%20strategies%20based%20on%20reinforcement%20learning.pdf)   [BibTex](MetaBBO/MetaBBO-RL/Differential%20Evolution/RL-HPSDE/BibTex)|
|DE-DQN|Tan, Zhiping, and Kangshun Li. "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing 111 (2021): 107678.|-|[PDF](MetaBBO/MetaBBO-RL/Differential%20Evolution/DE-DQN/Differential%20evolution%20with%20mixed%20mutation%20strategy%20based%20on%20deep%20reinforcement%20learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Differential%20Evolution/DE-DQN/BibTex)|
|LDE|Sun, Jianyong, et al. "[**Learning Adaptive Differential Evolution Algorithm from Optimization Experiences by Policy Gradient**](https://ieeexplore.ieee.org/abstract/document/9359652)." IEEE Transactions on Evolutionary Computation 25.4 (2021): 666-680.|[yierh/LDE](https://github.com/yierh/LDE)|[PDF](MetaBBO/MetaBBO-RL/Differential%20Evolution/LDE/Learning%20Adaptive%20Differential%20Evolution%20Algorithm%20from%20Optimization%20Experiences%20by%20Policy%20Gradient.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Differential%20Evolution/LDE/BibTex)|
|*MARLwCMA|Sallam, Karam M., et al. "[**Evolutionary framework with reinforcement learning-based mutation adaptation**](https://ieeexplore.ieee.org/abstract/document/9239320/)." IEEE Access 8 (2020): 194045-194071.|-|[PDF](MetaBBO/MetaBBO-RL/Differential%20Evolution/MARLwCMA/Evolutionary%20framework%20with%20reinforcement%20learning-based%20mutation%20adaptation.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Differential%20Evolution/MARLwCMA/BibTex)|
|DE-DDQN|Sharma, Mudita, et al. "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference. 2019.|[mudita11/DE-DDQN](https://github.com/mudita11/DE-DDQN)|[PDF](MetaBBO/MetaBBO-RL/Differential%20Evolution/DE-DDQN/Deep%20reinforcement%20learning%20based%20parameter%20control%20in%20differential%20evolution.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Differential%20Evolution/DE-DDQN/BibTex)|
|*DE-RLFR|Li, Zhihui, et al. "[**Differential evolution based on reinforcement learning with fitness ranking for solving multimodal multiobjective problems**](https://www.sciencedirect.com/science/article/pii/S2210650218310575)." Swarm and Evolutionary Computation 49 (2019): 234-244.|-|[PDF](MetaBBO/MetaBBO-RL/Differential%20Evolution/DE-RLFR/Differential%20evolution%20based%20on%20reinforcement%20learning%20with%20fitness%20ranking%20for%20solving%20multimodal%20multiobjective%20problems.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Differential%20Evolution/DE-RLFR/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

#### 3.1.2. Partical Swarm Optimization

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|RL-PSO|Wu, Di, and G. Gary Wang. "[**Employing reinforcement learning to enhance particle swarm optimization methods**](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1867120)." Engineering Optimization 54.2 (2022): 329-348.|-|[PDF](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/RL-PSO/Employing%20reinforcement%20learning%20to%20enhance%20particle%20swarm%20optimization%20methods.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/RL-PSO/BibTex)|
|RLEPSO|Yin, Shiyuan, et al. "[**RLEPSO: Reinforcement learning based Ensemble particle swarm optimizer**](https://dl.acm.org/doi/abs/10.1145/3508546.3508599)." Proceedings of the 2021 4th International Conference on Algorithms, Computing and Artificial Intelligence. 2021.|-|[PDF](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/RLEPSO/RLEPSO%20Reinforcement%20learning%20based%20Ensemble%20particle%20swarm%20optimizer.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/RLEPSO/BibTex)|
|QLPSO|Xu, Yue, and Dechang Pi. "[**A reinforcement learning-based communication topology in particle swarm optimization**](https://link.springer.com/article/10.1007/s00521-019-04527-9)." Neural Computing and Applications 32 (2020): 10007-10032.|-|[PDF](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/QLPSO/A%20reinforcement%20learning-based%20communication%20topology%20in%20particle%20swarm%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/QLPSO/BibTex)|
|*QLSOPSO \& QLMOPSO|Liu, Yaxian, et al. "[**An adaptive online parameter control algorithm for particle swarm optimization based on reinforcement learning**](https://ieeexplore.ieee.org/abstract/document/8790035/)." 2019 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2019.|-|[PDF](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/QLSOPSO\&QLMOPSO/An%20adaptive%20online%20parameter%20control%20algorithm%20for%20particle%20swarm%20optimization%20based%20on%20reinforcement%20learning.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/QLSOPSO\&QLMOPSO/BibTex)|
|*RLMPSO|Samma, Hussein, Chee Peng Lim, and Junita Mohamad Saleh. "[**A new reinforcement learning-based memetic particle swarm optimizer**](https://www.sciencedirect.com/science/article/pii/S1568494616000132)." Applied Soft Computing 43 (2016): 276-297.|-|[PDF](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/RLMPSO/A%20new%20reinforcement%20learning-based%20memetic%20particle%20swarm%20optimizer.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Partical%20Swarm%20Optimization/RLMPSO/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

#### 3.1.3. Others

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|*MADAC|Xue, Ke, et al. "[**Multi-agent dynamic algorithm configuration**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022): 20147-20161.|-|[PDF](MetaBBO/MetaBBO-RL/Others/MADAC/Multi-agent%20dynamic%20algorithm%20configuration.pdf)  [BibTex](MetaBBO/MetaBBO-RL/Others/MADAC/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 3.2. MetaBBO with Supervised Learning (MetaBBO-SL)

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|LTO-POMDP|Gomes, Hugo Siqueira, Benjamin Léger, and Christian Gagné. "[**Meta learning black-box population-based optimizers**](https://arxiv.org/abs/2103.03526)." arXiv preprint arXiv:2103.03526 (2021).|[LTO-POMDP](https://github.com/optimization-toolbox/meta-learning-population-based-optimizers)|[PDF](MetaBBO/MetaBBO-SL/LTO-POMDP/Meta%20learning%20black-box%20population-based%20optimizers.pdf)  [BibTex](MetaBBO/MetaBBO-SL/LTO-POMDP/BibTex)|
|RNN-Opt|TV, Vishnu, et al. "[**Meta-learning for black-box optimization**](http://proceedings.mlr.press/v70/chen17e.html)." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer International Publishing, 2019.|-|[PDF](MetaBBO/MetaBBO-SL/RNN-Opt/Meta-learning%20for%20black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SL/RNN-Opt/BibTex)|
|RNN-OI|Chen, Yutian, et al. "[**Learning to learn without gradient descent by gradient descent**](http://proceedings.mlr.press/v70/chen17e.html)." International Conference on Machine Learning. PMLR, 2017.|-|[PDF](MetaBBO/MetaBBO-SL/RNN-OI/Learning%20to%20learn%20without%20gradient%20descent%20by%20gradient%20descent.pdf)  [BibTex](MetaBBO/MetaBBO-SL/RNN-OI/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 3.3. MetaBBO with Self-Referential Search (MetaBBO-SR)

|Algorithm|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
|LES|Lange, Robert, et al. "[**Discovering evolution strategies via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583133.3595822)." Proceedings of the Companion Conference on Genetic and Evolutionary Computation. 2023.|-|[PDF](MetaBBO/MetaBBO-SR/LES/Discovering%20evolution%20strategies%20via%20meta-black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SR/LES/BibTex)|
|LGA|Lange, Robert, et al. "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. 2023.|-|[PDF](MetaBBO/MetaBBO-SR/LGA/Discovering%20attention-based%20genetic%20algorithms%20via%20meta-black-box%20optimization.pdf)  [BibTex](MetaBBO/MetaBBO-SR/LGA/BibTex)|

<p align="right">
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>

### 3.4. Other MetaBBO

## 4. Benchmarks

|Benchmark|Paper|Original Repository|About|
|:-:|:-:|:-:|:-:|
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
