# -*- coding: utf-8 -*-
"""
@author: Khalid Aram
Alternated Sorting Method Genetic Algorithm (ASMGA) 
A hybrid algorithm for feature and model selection for Support Vector Machine (SVM) classifiers
"""

from pylab import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
#from scikitplot.metrics import plot_roc
from scipy import stats as st
import time
import pygmo as pg
from collections import Counter
import pandas as pd
from tqdm import tqdm as td



def feat_values(X, y):# produces emperical dist based on feature distribution #add kernel type Q function
    """KS-MMFS Implemetation:
    Details are in the following paper: 
    "Aram, K. Y., Lam, S. S., & Khasawneh, M. T. (2022). 
    Linear Cost-sensitive Max-margin Embedded Feature Selection for SVM. 
    Expert Systems with Applications, 197, 116683."
    """
    a = [1./len(X[0]) for _ in range(len(X[0]))]
    r = []
    for j in range(len(X[0])):
        r.append((j,abs(dot(y,X[:,j]))))
    Q = zeros((len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if i == j: Q[i][j] = 0
#            else: Q[i][j] = dot(X[:,i],X[:,j])# linear kernel
            else: Q[i][j] = exp(-1*sum((X[:,i]-X[:,j])**2)) 
    redundancy = [(_, dot(a,Q[:,_])) for _ in range(len(a))]

    f_values = [r[_][1] - redundancy[_][1] for _ in range(len(a))] #to be maximized

    min_value = min(f_values)
    f_values = [_-min_value for _ in f_values] 
    return [float(_)/sum(f_values) for _ in f_values] #distribution




#FPR=1-specificity(TNR)
#FNR=1-sensitivity(TPR)


def scores(y_actual, y_predicted):
    """produces metrics for evaluating classification results: 
        False Positive Rate (FPR), False Negative Rate (FNR), Error Rate;
        classes: positive = 1, negative = -1
        0.0001 is added to denominators to avoid division by 0"""
    
    count = {(1,1):0., (1,-1):0., (-1,1):0., (-1,-1):0.} #count of TP,FN,FP,TN
    c = Counter(zip(y_actual, y_predicted))  
    for pair, freq in c.items(): count[pair] += freq
    return {"FNR": count[(1,-1)]/(0.0001+count[(1,1)] + count[(1,-1)]), 
                  "FPR": count[(-1,1)]/(0.0001+count[(-1,-1)] + count[(-1,1)]), 
                  "ERR": (count[(1,-1)]+count[(-1,1)])/sum(list(count.values()))}



class Solution:
    """Class for defining gentic algorithm chromosomes"""
    def __init__(self, sol_len, blank = False, values = []): 
        self.sol_len = sol_len
        self.sol = []
        self.feat_values = values
        
#        if not blank:
#            self.sol = [choice([0,1]) for _ in range(self.sol_len)]
        
        if not blank:
            if rand() <= 0.5: # random solutions for diversity
                self.sol = [choice([0,1]) for _ in range(self.sol_len)]
            else: #biased selection
                self.sol = [0 for _ in range(self.sol_len)]
                n_selected = choice(range(1,self.sol_len))
                index = choice(range(self.sol_len), size = n_selected, replace=False, p = self.feat_values)
                for _ in index:
                    self.sol[_] = 1
            self.param = [rand()*32, rand()*32]
                                    
        else: 
            self.sol = [1 for _ in range(self.sol_len)]
            self.param = [1., 1./self.sol_len] #default kernel param [C, Y]
            
        self.ERR = 1.
        self.perc_selected = 1.
        self.features = range(self.sol_len)
        self.FPR = 1.
        self.FNR = 1.
        self.metrics = 1.
        self.sol_value = float(inf) # to be minimized (accounts for # feature selected and value of features)
        
    def selected(self): #Determines the selected features in the chromosome
        if sum(self.sol) == 0:
            self.features = list(range(self.sol_len))
            self.sol = [1 for _ in range(self.sol_len)]
            self.param = [1., 1./self.sol_len] #default kernel param [C, Y]
            self.perc_selected = 1.
            self.metrics = 1.
        else:##
            self.features = list(filter(lambda i : self.sol[i] == 1, range(self.sol_len)))

            self.perc_selected = len(self.features)/float(self.sol_len) # percentage of features selected
        
    def crossover(self, sol2 = None): #GA crossover function
        for _ in range(self.sol_len):
            temp = self.sol[_]
            if rand() <= 0.5:
                self.sol[_] = sol2.sol[_]
                sol2.sol[_] = temp

        
        for _ in range(2):
            pars1 = self.param
            pars2 = sol2.param
            rn0, rn1 = rand(), rand()
            # C crossover
            self.param[0] = rn0*pars1[0] + (1-rn0)*pars2[0]
            sol2.param[0] = (1-rn0)*pars1[0] + rn0*pars2[0]
            #gamma crossover
            self.param[1] = rn1*pars1[1] + (1-rn1)*pars2[1]
            sol2.param[1] = (1-rn1)*pars1[1] + rn1*pars2[1]
        
        self.param = list(clip(self.param, 0.0001, 32))
        sol2.param = list(clip(sol2.param,0.0001, 32))
        del temp, rn0, rn1
        
    def mutation(self): #GA mutation function
        pm = 1./(2+self.sol_len) ## document pm
        def flip_(a): 
            return 1 - a if rand()<=pm else a

        self.sol = list(map(flip_, self.sol))
        
        # mutate C and gamma
        if rand()<pm: self.param[0] = self.param[0] + normal(0, 1)
        if rand()<pm: self.param[1] = self.param[1] + normal(0, 1)
        self.param = list(clip(self.param, 0.0001, 32))
        
        

    def evaluate(self, X, y): #Evaluates the chromosom based on the selected features and parameters
        self.selected()

        self.sol_value = (self.perc_selected)+ 1/(1+dot(self.sol, self.feat_values)) #to be minimized

        if sum(self.sol) > 0:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            clf = SVC(kernel ='rbf', class_weight='balanced', C = self.param[0], gamma = self.param[1]/(x_train.var()*self.sol_len))
            clf.fit(x_train[:, self.features], y_train)
            y_predicted = clf.predict(x_test[:, self.features])
            score = scores(y_test, y_predicted)
            self.FNR, self.FPR, self.ERR = score['FNR'], score['FPR'], score['ERR']   
            
            counts = Counter(y)


            wFNR, wFPR = 1-counts[1]/len(y), 1-counts[-1]/len(y)
            
            self.metrics = (wFPR*self.FPR + wFNR*self.FNR)
       
            
    def __str__(self): #Prints chromosome info
        return "".join(["Accuracy: {}".format(str(round(1-self.ERR, 4))),
                        " Sensitivity: {}".format(str(round(1-self.FNR, 4))),
                        " Specificity: {}".format(str(round(1-self.FPR, 4))),
                        " %Selected: {}%".format(str(round(100*self.perc_selected, 4)))])       
class ASMGA:
    """ASMGA Search Instance"""
    def __init__(self, X = [], y = [], pop_size = 50, k = 3, gen = 100, plot = True, values=[]):
        self.plot = plot
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.k = k
        self.gen = gen
        self.sol_len = len(X[0])
        self.pop = []
        self.best = None
        self.search_log = []
        self.values=values

    def plot_(self):
        figure("ASMGA Search")
        xlabel("Generation")
        ylabel("Objective Value")
        title("ASMGA Search Log\n-Best Fitness-")
        plot(range(self.gen), array(self.search_log))   
        
        
    def gen_sol(self, blank = False, values = []):#generates a chromosome
        return Solution(self.sol_len, blank = blank, values = values)
    
    def select(self, method = "mo"): # tournament selection for GA, k = 3
        tournament  = [choice(self.pop) for _ in range(self.k)]
        if method == "mo":#specify selection method (single objective (so) or multo-objective (mo))       
            selected = pg.select_best_N_mo(points = [[_.metrics, _.sol_value] for _ in tournament], N = 1)
            return tournament[selected[0]]
        else:
            tournament = sorted(tournament, key = lambda individual: individual.metrics+individual.sol_value)
            return tournament[0]
        
        selected, tournament = None, None 
   
    def sort(self, method = "mo"):#sorts chromosoms based on multi-objective performance
        if method == "mo": 
            obj_vectors = [[_.metrics, _.sol_value] for _ in self.pop]
            sorting = pg.sort_population_mo(points = obj_vectors)
            self.pop = [self.pop[_] for _ in sorting]
        else:
            self.pop = sorted(self.pop, key = lambda individual: individual.metrics+individual.sol_value)
        obj_vectors, sorting = None, None

        
    def update_best(self, method = "mo"):#updates the best solution found so far
        if method == "mo": 
            obj_vectors = [[_.metrics, _.sol_value] for _ in self.pop]
            sorting = pg.sort_population_mo(points = obj_vectors)
            best = self.pop[sorting[0]]
            self.best.__dict__.update(best.__dict__)


        else:
            pop1 = self.pop
            pop1 = sorted(pop1, key = lambda individual: individual.metrics+individual.sol_value)
            self.best.__dict__.update(pop1[0].__dict__)
         
                
    def initiate_pop(self): #initiates GA population

        self.pop = [self.gen_sol(values = self.values) for _ in range(self.pop_size)]
        for s in self.pop: s.evaluate(self.X, self.y)
        self.best = self.gen_sol(blank = True)

 
    def breed(self, method): #applies crossover and mutation on selected parents
        for _ in range(int(self.pop_size/2.)):            
            child1 = self.gen_sol(blank = True)
            child2 = self.gen_sol(blank = True)
            parent1 = self.select(method=method)
            parent2 = self.select(method=method)
            
            child1.__dict__.update(parent1.__dict__)
            child2.__dict__.update(parent2.__dict__)

            child1.crossover(sol2 = child2)
            child1.mutation()
            child2.mutation()
            #evaluate only new members, elitist approach is faster
            child1.evaluate(self.X, self.y)
            child2.evaluate(self.X, self.y)
            
            self.pop += [child1, child2]                

    def elite(self, method): #performs elitist selection of next population
        self.sort(method=method)
        for s in self.pop[self.pop_size:]: del s
        self.pop = self.pop[0: self.pop_size] #Elitism
        self.search_log.append([self.pop[0].metrics, self.pop[0].sol_value])
        
    def run(self):#runs ASMGA

        self.initiate_pop()

        self.search_log = []
       
        methods=["so" for _ in range(int(0.5*self.gen))]+["mo" for _ in range(int(0.5*self.gen))]   
        
        for i in td(range(self.gen)):  #ASMGA search
#            print(i)
#            print("\rProgress: {}%".format(round(100*i/self.gen)), end='', flush=True )
            self.breed(methods[i])
            self.elite(methods[i])
        
        #final result:
        self.update_best(method='mo')
        if self.plot: self.plot_() # plots search log
        
        '''Pareto Fronts'''
        pg.plot_non_dominated_fronts([[_.FNR, _.FPR, _.sol_value] for _ in self.pop])
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting([[_.FNR, _.FPR, _.sol_value] for _ in self.pop])
        return pd.DataFrame({"FNR":[s.FNR for s in self.pop], 
                "FPR":[s.FPR for s in self.pop],
                "Metrics":[s.metrics for s in self.pop],
                "%Selected":[s.perc_selected for s in self.pop],
                "Front":list(ndr)})
        
        return self.best

def cross_val(model):
    '''produces 5-fold cross-validation results replicated 10 times'''
    
    X = model.X
    y = model.y
    
    Sens_reps = []
    Spec_reps = []
    Gmean_reps = []
    Acc_reps = []
    perc_reps = []
    stabs_reps = []
    times_reps = []
    
    for i in range(10): #number of runs/replicates
        print('Replicate {}'.format(i))
        seed(i)
        # Cross-validation results for each replicate:
        Sens = []
        Spec = []
        Gmean = []
        Acc = []
        perc = []
        sols = []
        t = time.time()
        
        folds = StratifiedKFold(n_splits=3)
        for train, test in folds.split(X, y):
            print("Fold")
            model.X = X[train]
            model.y = y[train]
            x_test = X[test]
            y_test = y[test]
            
            solution = model.run() #select features
            
            clf = SVC(kernel ='rbf',class_weight='balanced', C = solution.param[0], gamma = solution.param[1]/(model.X.var()*len(X[0])))
            clf.fit(model.X[:, solution.features], model.y) #train using selected features 
            y_predicted = clf.predict(x_test[:, solution.features]) #testing on unforseen data using selected features
            score = scores(y_test, y_predicted) # test scores
            Sens.append(1 - score["FNR"])
            Spec.append(1 - score["FPR"])
            Gmean.append(sqrt((1 - score["FNR"])*(1 - score["FPR"])))
            Acc.append(1 - score["ERR"])
            perc.append(solution.perc_selected)
            sols.append(solution.features)
            
        # Aggregate results accross replicates:
        times_reps.append(time.time()-t)
        Sens_reps.append(mean(Sens))
        Spec_reps.append(mean(Spec)) 
        Gmean_reps.append(mean(Gmean))
        Acc_reps.append(mean(Acc))
        perc_reps.append(mean(perc))
        stabs_reps.append(stability(sols))
        
        
    return pd.DataFrame({'measure':['AVG', 'STDEV', 'CI'], 'Sens':[mean(Sens_reps), std(Sens_reps, ddof=1), conf_int(Sens_reps)], 
            'Spec':[mean(Spec_reps), std(Spec_reps, ddof=1), conf_int(Spec_reps)], 
            'Gmean':[mean(Gmean_reps), std(Gmean_reps, ddof=1), conf_int(Gmean_reps)],
            'Acc':[mean(Acc_reps), std(Acc_reps, ddof=1), conf_int(Acc_reps)],
            '%Selected':[mean(perc_reps), std(perc_reps, ddof=1), conf_int(perc_reps)],
            'Stab':[mean(stabs_reps), std(stabs_reps, ddof=1), conf_int(stabs_reps)],
            'Time': [mean(times_reps), std(times_reps, ddof=1), conf_int(times_reps)]})



def roc_curve(x, y, model = SVC(kernel ='linear')):
    ''''plots roc curve of fitted model'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model.fit(x_train, y_train)
    y_pred_pr = model.predict_proba(x_test)
    plot_roc(y_test, y_pred_pr)

def conf_int(data):
    """calculates a confidence interval for a given list of numbers"""
    interval = st.t.interval(0.95, len(data)-1, loc=mean(data), scale=st.sem(data))
    return (interval[1] - interval[0])/2.

def stability(sols):
    '''returns similarity of a set of solutions (selected feature sets)'''
    def similarity(s1, s2):
        '''similarity measure: # common elements/#distinct elements'''
        return len(set(s1) & set(s2))/float(len(set(s1) | set(s2)))
    sims = []
    for i in range(len(sols)-1):
        for j in range(i+1, len(sols)):
            sims.append(similarity(sols[i], sols[j]))
    return  mean(sims)

"""Model Building and Evaluation"""
   
def main():
    
    dataset_names = ['Leukemia', 'Mfeat', 'Musk', 'Biodeg', 'Chess', 'Yeast', 
                     'Housing', 'Isolet', 'Hepatitis', 'Heart Failure', 'SPECT Heart']
    datasets = ['Leukemia.txt', 'Mfeat.txt', 'Musk.txt',
                'Biodeg.txt', 'Chess.txt', 'Yeast.txt', 'Housing.txt', 
                'Isolet.txt', 'Hepatitis.txt', 'Heart_Failure.txt', 
                'SPECT_Heart.txt']

    pop_sizes = [200, 75, 50, 50, 50, 50, 75]
    gens = [400, 150, 100, 100, 100, 100, 150]#152 used to avoid fractions in sorting method schedule
    
    RESULTS = pd.DataFrame(columns = ['dataset', 'measure', 'Sens',
            'Spec',
            'Gmean',
            'Acc',
            '%Selected',
            'Stab',
            'Time'])
    RES = []    
    for i in [5]:
        print('Working on {}'.format(dataset_names[i]))
        f = loadtxt(datasets[i])

        shuffle(f)
        X = f[:,0:-1]
        y = f[:,-1]
        values = feat_values(X,y)
        model = ASMGA(X, y, pop_size=pop_sizes[i], gen=gens[i], plot=False, values=values)
        res = cross_val(model)
        res['dataset'] = [dataset_names[i] for _ in range(3)]#edit
        res.to_csv("ASMGA_RES.csv", index=False, mode='a')
        RES.append(res)

    RESULTS = pd.concat(RES)
    print(RESULTS)
        
if __name__ == '__main__':
    main()
