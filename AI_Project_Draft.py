import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import datasets
from itertools import combinations
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from statistics import mode, mean
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
import time
import os
import psutil
import random as rd
import statistics
import heapq
import tracemalloc as tcm


data_X, data_y = datasets.load_breast_cancer(return_X_y= True)
model = DecisionTreeClassifier(max_depth=4)
model.fit(data_X, data_y)
model_feature_importance = model.feature_importances_
mean_value = statistics.mean(model_feature_importance)

print("The mean value of feature importance is: ", mean_value)
for i in range(0, len(model_feature_importance)):
    if model_feature_importance[i] < mean_value:
        model_feature_importance[i] = 0

array_of_features = []
for i in range(0, len(model_feature_importance)):
    if not model_feature_importance[i] == 0:
        array_of_features.append(i)

print("Best combination of features based on feature selection by decision tree: ", array_of_features)
print()
# Finding  out best features.


verifying_X = data_X[-100:-99]
verifying_y = data_y[-100:-99]
data_X = data_X[:, array_of_features]
unseen_test_data_X = data_X[-99:]
unseen_test_data_y = data_y[-99:]
verifying_X = data_X[-100:-99]
verifying_y = data_y[-100:-99]
data_X = data_X[:-100]
data_y = data_y[:-100]




kf = KFold(n_splits=4)
z = kf.get_n_splits(data_X)
k_values = [3, 5, 7, 9, 11, 13]
MSE_array = []

for i in k_values:
    classifier = KNeighborsClassifier(n_neighbors = i) # create the model - reserve space and initialize
    sum = 0
    for train_index, test_index in kf.split(data_X):
        X_train, X_validate = data_X[train_index], data_X[test_index]
        y_train, y_validate = data_y[train_index], data_y[test_index]
        classifier.fit(X_train,y_train) #training
        y_pred  = classifier.predict(X_validate) #testing
        count = 0
        for k in range(0, len(y_pred)):
            if not y_pred[k] == y_validate[k]:
                count += 1
        sum += count
    MSE = sum ** 2
    MSE_array.append(MSE/4)

hyper_k = k_values[MSE_array.index(min(MSE_array))]
print('Optimised hyperparameter K is: ', hyper_k)
print("This is the Mean Squared Error (MSE) array for each K value: ", MSE_array)
print()
# Optimising hyperparameter K.


counting = []
for j in range(0, 100):
    depth_values = [3, 4, 5, 6]
    MSE_array = []
    for i in depth_values:
        classifier = DecisionTreeClassifier(max_depth = i, splitter="best") # create the model - reserve space and initialize
        sum = 0
        for train_index, test_index in kf.split(data_X):
            X_train, X_validate = data_X[train_index], data_X[test_index]
            y_train, y_validate = data_y[train_index], data_y[test_index]
            classifier.fit(X_train,y_train) #training
            y_pred  = classifier.predict(X_validate) #testing
            count = 0
            for k in range(0, len(y_pred)):
                if not y_pred[k] == y_validate[k]:
                    count += 1
            sum += count
        MSE = sum ** 2
        MSE_array.append(MSE/4)
    x = depth_values[MSE_array.index(min(MSE_array))]
    counting.append(x)

hyper_depth = mode(counting)
print("Optimised hyperparameter depth is: ", hyper_depth)
print("This is the Mean Squared Error (MSE) array for each depth value: ", MSE_array)
# Optimising hyperparameter depth.



ensemble_classifier = VotingClassifier(estimators=[('NN', MLPClassifier(alpha=1, max_iter=1000)),
                                                   ('AdaBoost', AdaBoostClassifier())
                                                   ,('Naive Baise', GaussianNB())
                                                   ,('Decision Tree',
                                                     DecisionTreeClassifier(max_depth = hyper_depth, splitter="best"))
                                                   ], voting='soft')
classifiers = [
    KNeighborsClassifier(hyper_k),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth = hyper_depth, splitter="best"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    ensemble_classifier]


Accuracy_array = []
for classifier in classifiers:
    sum = 0
    for train_index, test_index in kf.split(data_X):
        X_train, X_validate = data_X[train_index], data_X[test_index]
        y_train, y_validate = data_y[train_index], data_y[test_index]
        classifier.fit(X_train,y_train) #training
        y_pred  = classifier.predict(X_validate) #testing
        count = 0
        for k in range(0, len(y_pred)):
            if y_pred[k] == y_validate[k]:
                count += 1
        sum += count
    Accuracy_array.append(round(100*(sum/len(data_y)), 2))
print()
print("Cross validation accuracy array for all classifiers: ", Accuracy_array)

recall_array = []
accuracy_array = []
precision_array = []
f_measure_array = []
execution_array = []
memory_array = []
roc_auc_array = []
classifiers_array = ["kNN", "Linear_SVM", "RBF_SVM", "Guassian", "Dec_Tree", "Rand_Forest",
                     "NN", "AdaBoost", "Naive Baise", "QDA", "Ensemble"]
z = ['red', 'green', 'yellow', 'blue', 'orange', 'pink', 'purple', 'navy', 'brown', 'black', 'magenta']
i = 0
for classifier in classifiers:
    tcm.start()
    start_time = time.time()
    classifier.fit(data_X, data_y)
    y_pred = classifier.predict(unseen_test_data_X)
    memoryval, notImportant = tcm.get_traced_memory()
    probs = classifier.predict_proba(unseen_test_data_X)
    preds = probs[:,[1]]
    execution_array.append(round((time.time() - start_time),3))
    memory_array.append(round(memoryval/1024, 3))
    tn, fp, fn, tp = confusion_matrix(unseen_test_data_y, y_pred).ravel()
    recall_array.append(round(tp/(tp+fn)*100, 3))
    accuracy_array.append(round(((tp+tn)/(tp+tn+fp+fn))*100, 3))
    precision_array.append(round(tp/(tp+fp)*100, 3))
    f_measure_array.append(round((2*tp)/(2*tp+fn+fp)*100, 3))
    fpr, tpr, thresholds = roc_curve(unseen_test_data_y ,preds)
    roc_auc_array.append(round(auc(fpr, tpr), 3))
    plt.plot(fpr, tpr, label=classifiers_array[i], color = z[i])
    tcm.stop()
    i += 1

plt.style.use('dark_background')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.style.use('dark_background')
plt.legend()
plt.show()
print()
tableArray = ["Classifier", "Accuracy (%)", "Precision (%)", "Recall (%)", "F measure (%)", "Exec Time (s)",
              "Memeroy Used (KB)", "AUC"]
data = [tableArray] + list(zip(classifiers_array, accuracy_array, precision_array, recall_array,
                               f_measure_array, execution_array, memory_array, roc_auc_array))
for i, d in enumerate(data):
    line = '|'.join(str(x).ljust(12) for x in d)
    print(line)
    if i == 0:
        print('-' * len(line))

highest_values = ["Highest Accuracy", "Highest Recall", "Highest Precision", "Highest F Measure"
    , "Lowest Execution Time", "Lowest Memory", "Highest Area Under ROC Curve"]
scores_array = [max(accuracy_array), max(recall_array), max(precision_array), max(f_measure_array)
                , min(execution_array), min(memory_array), max(roc_auc_array)]
classTable = [classifiers_array[accuracy_array.index(max(accuracy_array))],
              classifiers_array[recall_array.index(max(recall_array))]
              , classifiers_array[precision_array.index(max(precision_array))],
              classifiers_array[f_measure_array.index(max(f_measure_array))]
              , classifiers_array[execution_array.index(min(execution_array))],
              classifiers_array[memory_array.index(min(memory_array))]
              , classifiers_array[roc_auc_array.index(max(roc_auc_array))]]
perc_and_time = ["%", "%", "%", "%", "s", "KB", ""]
for i in range(0, len(highest_values)):
    print(highest_values[i], " is: ", scores_array[i],"", perc_and_time[i], " for classifier: ", classTable[i])



def showBarChart(classifier, accuracy, title, x, y):
    z = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple']
    plt.style.use('dark_background')
    plt.bar(classifier, accuracy, color=z[p])
    plt.suptitle(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation='vertical')
    plt.autoscale()
    plt.tight_layout()
    plt.show()

title_array = ['Accuracy vs Classifiers Plotting',
             'Precision vs Classifiers Plotting',
                'Recall vs Classifiers Plotting',
                'F measure vs Classifiers Plotting',
                'Execution time vs Classifiers Plotting',
                'Memory usage vs Classifiers Plotting',
                'Area Under ROC Curve vs Classifiers Plotting']
Reporting_Array = [accuracy_array, precision_array, recall_array, f_measure_array, execution_array,
                   memory_array, roc_auc_array]
x = 'Classifiers'
y = ['Accuracy','Precision','Recall','F measure','Execution time (s)','Memory used in KB','Area' ]
p=0
for n in range(0,len(Reporting_Array)):
    title = title_array[n]
    X = x
    Y = y[n]
    showBarChart(classifiers_array, Reporting_Array[n],title,X,Y)
    p+=1


comparison_accuracy_array = []
for i in range(0, 30):
    accuracy_array = []
    data_X, data_y = datasets.load_breast_cancer(return_X_y= True)
    data_X = data_X[:, np.newaxis, i]
    unseen_test_data_X = data_X[-100:]
    unseen_test_data_y = data_y[-100:]
    data_X = data_X[:-100]
    data_y = data_y[:-100]
    for classifier in classifiers:
        classifier.fit(data_X, data_y)
        y_pred = classifier.predict(unseen_test_data_X)
        tn, fp, fn, tp = confusion_matrix(unseen_test_data_y, y_pred).ravel()
        accuracy_array.append(round(tp+tn/(tp+tn+fp+fn)*100))
    comparison_accuracy_array.append(mean(accuracy_array))

Vernua_features_array =[]
best_pred1 = comparison_accuracy_array.index(max(comparison_accuracy_array))
comparison_accuracy_array.pop(best_pred1)
best_pred2 = comparison_accuracy_array.index(max(comparison_accuracy_array))
if best_pred2 > best_pred1:
    best_pred2 -= 1
elif best_pred2 <= best_pred1:
    best_pred2 += 1
Vernua_features_array.append(best_pred1)
Vernua_features_array.append(best_pred2)
print("The best two predictors based on accuracy are: ", Vernua_features_array)



data_X, data_y = datasets.load_breast_cancer(return_X_y= True)
data_X= data_X[:,(Vernua_features_array)]
h = .02  # step size in the mesh
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
datasets = [(data_X, data_y)]
figure = plt.figure(figsize=(37, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    # iterate over classifiers
    for name, clf in zip(classifiers_array, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1
plt.tight_layout()
plt.show()
