# ASMGA: Alternated Sorting Method Genetic Algorithm
**Language: Python 2.7**

ASMGA is hybrid wrapper-filter algorithm for feature and model selection for Support Vector Machine (SVM) classifiers.
Three objectives are considered: cost-sensitive error rate, feature subset size, and feature importance in terms of relevance and redundancy. 
The algorithm approximates a set of Pareto optimal solutions. A solution includes selected features and values of RBF SVM parameters. 
ASMGA hybrid uses two multi-objective sorting techniques: Weighted Sum (WS) of objectives and Non-dominated Sorting (NDS) in the breeding process and for selecting surviving solutions in each GA generation. 
Knapsack Max-margin Feature Selection (KS-MMFS) objective function coefficients (Aram et al., 2022), are used as the second objective function.
KS-MMFS provide estimates of feature importance based on relevance and redundancy for the features in the selected subset. 
A mix of random and biased solutions is used in the initial population. 
ASMGA includes a strategy that alternates between WS and NDS. Thus, the algorithm works as elitist GA for some generations and as a Non-dominated Sorting Genetic Algorithm (NSGA-II) for the remaining generations. 
This strategy relies on a schedule of sorting methods, by which, a user can define deploy different sorting methods per different sequences of generations. 
The chart below shows the main steps of the proposed ASMGA. 

![image](https://user-images.githubusercontent.com/57454095/172543282-334c7c50-1892-4e57-910a-1760a939ed1c.png)


References:

Aram, K. Y., Lam, S. S., & Khasawneh, M. T. (2022). Linear Cost-sensitive Max-margin Embedded Feature Selection for SVM. Expert Systems with Applications, 197, 116683.
