
import json
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import time

plt.style.use('ggplot')


def define_clfs_params(grid_size):

    classifiers = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return classifiers, large_grid
    elif (grid_size == 'small'):
        return classifiers, small_grid
    elif (grid_size == 'test'):
        return classifiers, test_grid
    else:
        return 0, 0
    
    
def classify(X, y, models, iters, threshold, metrics,classifiers,grid):
    '''
    Takes:
        X, a dataframe of features 
        y, a dataframe of the label
        models, a list of strings indicating models to run (e.g. ['LR', 'DT'])

    Returns:
        A new dataframe comparing each classifier's performace on the given
        evaluation metrics.
    '''

    all_models = {}

    # Construct train and test splits
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # for every classifier, try any possible combination of parameters on grid
    for index, clf in enumerate([classifiers[x] for x in models]):
        name = models[index]
        print(name)
        parameter_values = grid[name]
        all_models[name] = {}
        
        # run the model with all combinations of the above parameters
        for p in ParameterGrid(parameter_values):
            precision_per_iter = []
            recall_per_iter = []
            f1_per_iter = []
            auc_per_iter = []
            time_per_iter = []
            avg_metrics = {}
            all_models[name][str(p)] = {}
            results = all_models[name][str(p)]
            clf.set_params(**p)

            # run iter number of iterations
            for i in range(iters): 
                try:
                    start_time = time.time()
                    
                    # get the predicted results from the model
                    if hasattr(clf, 'predict_proba'):
                        yscores = clf.fit(xtrain, ytrain).predict_proba(xtest)[:,1]
                    else:
                        yscores = clf.fit(xtrain, ytrain).decision_function(xtest)
                    yhat = np.asarray([1 if i >= threshold else 0 for i in yscores])
                    end_time = time.time()

                    # obtain metrics
                    mtrs = evaluate_classifier(ytest, yhat)
                    for met, value in mtrs.items():
                        eval('{}_per_iter'.format(met)).append(value)
                    time_per_iter.append(end_time - start_time)
                    print(end_time - start_time)
                    
                except IndexError:
                    print('Error')
                    continue
            
            avg_metrics['time'] = np.mean(time_per_iter)
            results['time'] = np.mean(time_per_iter)
            

            # store average metrics of model p
            for met in metrics:
                avg_metrics[met] = np.mean(eval('{}_per_iter'.format(met)))
                results[met] = avg_metrics[met]

        print('Finished running {}'.format(name))

    # dump everything in a json for future reference
    with open('all_models.json', 'w') as fp:
        json.dump(all_models, fp)

    return all_models

def select_best_models(results, models, d_metric):
    columns = ['auc', 'f1', 'precision', 'recall', 'time', 'parameters']
    rv = pd.DataFrame(index = models, columns = columns)
    best_metric = 0
    best_models = {}

    for model, iters in results.items():
        top_intra_metric = 0
        best_models[model] = {}
        for params, metrics in iters.items():
            header = [key for key in metrics.keys()]
            if metrics[d_metric] > top_intra_metric:
                top_intra_metric = metrics[d_metric]
                best_models[model]['parameters'] = params
                best_models[model]['metrics'] = metrics

        to_append = [value for value in best_models[model]['metrics'].values()]
        to_append.append(best_models[model]['parameters'])
        rv.loc[model] = to_append
        if top_intra_metric > best_metric:
            best_metric = top_intra_metric
            best_model = model, params

    return rv, best_models, (best_model, best_metric)

def gen_precision_recall_plots(X, y, best_models,classifiers):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

    for name, d in best_models.items():
        clf = classifiers[name]
        p = eval(d['parameters'])
        clf.set_params(**p)
        y_true = ytest
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.fit(xtrain, ytrain).predict_proba(xtest)[:,1]
        else:
            y_prob = clf.fit(xtrain, ytrain).decision_function(xtest)
        plot_precision_recall_n(y_true, y_prob, name)

def evaluate_classifier(ytest, yhat):
    '''
    For an index of a given classifier, evaluate it by various metrics
    '''
    # Metrics to evaluate
    metrics = {'precision': precision_score(ytest, yhat),
                'recall': recall_score(ytest, yhat),
                'f1': f1_score(ytest, yhat),
                'auc': roc_auc_score(ytest, yhat)}
    
    return metrics

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    plt.savefig(name)
    plt.show()

