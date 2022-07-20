# ASMGA: Alternated Sorting Method Genetic Algorithm

**Language:** Python 3.7

**Contents:**
 - Main ASMGA code
 - Datasets (text files)

ASMGA is a multi-objective hybrid wrapper-filter algorithm for feature and model selection for binary Support Vector Machine (SVM) classifiers. Theoritical background and detailed development of ASMGA can be found in Aram (2021).
ASMGA hybrid uses two multi-objective sorting techniques: Weighted Sum (WS) of objectives and Non-dominated Sorting (NDS) in the breeding process. 
 
ASMGA includes a strategy that alternates between WS and NDS. Thus, the algorithm works as elitist GA for some generations and as a Non-dominated Sorting Genetic Algorithm (NSGA-II) for the remaining generations. 
 
The chart below shows the main steps of ASMGA.

![image](https://user-images.githubusercontent.com/57454095/172543282-334c7c50-1892-4e57-910a-1760a939ed1c.png)


**References:**

Aram, K. Y. A. (2021). Max-margin cost-sensitive feature selection for support vector machines (Order No. 28542293). Available from ProQuest Dissertations & Theses Global.

**Inquiries can be sent to:**

Khalid Aram, PhD

karam@emporia.edu
