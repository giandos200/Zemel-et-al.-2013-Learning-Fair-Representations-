# Zemel-et-al.-2013-Learning-Fair-Representations-
Reproduction of Zemel, et al. ["Learning Fair Representations"](http://proceedings.mlr.press/v28/zemel13.pdf), ICML 2013
The code take inspiration from two repositories.
- The author repo: https://github.com/zjelveh/learning-fair-representations
- AIF360 repo: https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/lfr.py

The idea was to reproduce part of the experiment proposed in the article since no guidelines are available in both the repositories, and different problems trying to reproduce the article with both the repo.
In the litterature, this is considered as a PRE-PROCESSING fair representation algorithm. However, the authors propose LFR as both a PRE and IN-PROCESSING algorithm.
Zemel, et al. use different not usual metrics:
- <img src="https://latex.codecogs.com/svg.image?yACC=1-\frac{1}{2}\sum_{n=1}^{N}&space;\mid&space;y_n&space;-&space;\hat{y}_n&space;\mid" title="yACC=1-\frac{1}{N}\sum_{n=1}^{N} \mid y_n - \hat{y}_n \mid" />
- <img src="https://latex.codecogs.com/svg.image?yDiscrim&space;=&space;\mid&space;\frac{\sum_{n:s_n=1}\hat{y}_n}{\sum_{n:s_n=1}1}&space;-&space;\frac{\sum_{n:s_n=0}\hat{y}_n}{\sum_{n:s_n=0}1}&space;\mid" title="yDiscrim = \mid \frac{\sum_{n:s_n=1}\hat{y}_n}{\sum_{n:s_n=1}1} - \frac{\sum_{n:s_n=0}\hat{y}_n}{\sum_{n:s_n=0}1} \mid" />
- <img src="https://latex.codecogs.com/svg.image?yNN&space;=&space;1&space;-&space;\frac{1}{Nk}\sum_{n=1}^{N}&space;\mid&space;\hat{y}_n&space;-&space;\sum_{j&space;\in&space;kNN(x_n)}&space;\hat{y}_j&space;\mid" title="yNN = 1 - \frac{1}{Nk}\sum_{n=1}^{N} \mid \hat{y}_n - \sum_{j \in kNN(x_n)} \hat{y}_j \mid" />

Following, the metrics used in the plots proposed by myself:
- <img src="https://latex.codecogs.com/svg.image?ACC&space;=&space;\frac{TP&plus;TN}{TP&plus;TN&plus;FP&plus;FN}" title="ACC = \frac{TP+TN}{TP+TN+FP+FN}" />
- <img src="https://latex.codecogs.com/svg.image?DEO&space;=&space;\mid&space;TPrate_{privileged}&space;-&space;TPrate_{unprivileged}&space;\mid" title="DEO = \mid TPrate_{privileged} - TPrate_{unprivileged} \mid" />
- <img src="https://latex.codecogs.com/svg.image?DAO&space;=&space;\frac{\mid&space;FPrate_{privileged}-&space;FPrate_{unprivileged}\mid&plus;\mid&space;TPrate_{privileged}-TPrate_{unprivileged}\mid}{2}" title="DAO = \frac{\mid FPrate_{privileged}- FPrate_{unprivileged}\mid+\mid TPrate_{privileged}-TPrate_{unprivileged}\mid}{2}" />
- <img src="https://latex.codecogs.com/svg.image?F\_DEO&space;=&space;ACC\times(1-DEO)" title="F\_DEO = ACC\times(1-DEO)" />
- <img src="https://latex.codecogs.com/svg.image?F\_DAO&space;=&space;ACC\times(1-DAO)" title="F\_DAO = ACC\times(1-DAO)" />

### Installation guidelines:
Following, the instruction to install the correct packages for running the experiments (numba==0.48.0 is mandatory)

```bash
$ python3 -m venv venv_lfr
$ source venv_lfr/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

### Training and test the model
To train and evaluate LFR with all the metrics, you may run the following command:

```bash
$ python -u main.py
```
or run the personilized dataset, changing LINE 19 & 20 ("sensitive_features" and "dataset") and the "dataloader" of main.py script with an IDE or with `sudo nano main.py`
### Results

Following the graphics of train and test at various thresholds for different metrics (Accuracy, Difference in Equal Opportunity, Difference in Averages Odds, Fair_DEO==ACC*(1-DEO), Fair_DAO==ACC*(1-DAO)) and both German and Adult.
The plot show the accuracy-fairness trade-off at various thresholdes for the generated(or predicted) y_hat.

- GERMAN dataset

![image](https://user-images.githubusercontent.com/60853532/155762802-7fa20e15-c4be-4bf2-96be-45a0dbea9f04.png)
!![image](https://user-images.githubusercontent.com/60853532/155762857-10f37c20-ef0e-4ed2-a64c-9dc8a2550d91.png)

```
Next the metrics used by Zemel, et al. on the Test set
Accuracy: 0.6702218787753681
Discrimination: 0.03959704798033192
Consistency: 0.4259762244816738
```
- ADULT dataset

![image](https://user-images.githubusercontent.com/60853532/155762889-0200d172-98b3-4c26-bf0c-e9525aec5c58.png)
![image](https://user-images.githubusercontent.com/60853532/155762913-be6f2490-e515-4c40-8fb5-7b957cd6410b.png)

```
Next the metrics used by Zemel, et al. on the Test set
Accuracy: 0.6303079728728369
Discrimination: 0.001884131225272645
Consistency: 0.7975905194197657
```
