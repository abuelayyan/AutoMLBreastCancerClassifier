import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
import statistics


data_X, data_y = datasets.load_breast_cancer(return_X_y= True)
model = DecisionTreeClassifier(max_depth=4)
model.fit(data_X, data_y)
model_feature_importance = model.feature_importances_
mean_value = statistics.mean(model_feature_importance)
for i in range(0, len(model_feature_importance)):
    if model_feature_importance[i] < mean_value:
        model_feature_importance[i] = 0
array_of_features = []
for i in range(0, len(model_feature_importance)):
    if not model_feature_importance[i] == 0:
        array_of_features.append(i)
# Finding  out best features.


verifying_X = data_X[-100:-99]
verifying_y = data_y[-100:-99]
print(verifying_X)
print("The correct classification is:", verifying_y, "which is benign")
data_X = data_X[:, array_of_features]
unseen_test_data_X = data_X[-99:]
unseen_test_data_y = data_y[-99:]
verifying_X = data_X[-100:-99]
verifying_y = data_y[-100:-99]
data_X = data_X[:-100]
data_y = data_y[:-100]
hyper_k = 5
hyper_depth = 3


name = input("Please enter the patient's name: ")
age = input("Kindly enter the patient's age: ")
emirates_ID_number = input("Please enter patients Emirates ID: ")
insur_check = input("Is the patient going to pay cash or use insurance? (Kindly enter cash or insurance) ")
if insur_check == "insurance":
    insurance_number = input("Enter patient's insurance number: ")
phone_number = input("Enter the patients phone number: ")
print("Kindly enter the tumour cell attributes below: ")
data_set = np.array([])
radius_mean = float(input("radius (mean): "))
texture_mean = float(input("texture (mean): "))
perimeter_mean = float(input("perimeter (mean): "))
area_mean = float(input("area (mean): "))
smoothness_mean = float(input("smoothness (mean): "))
compactness_mean = float(input("compactness (mean): "))
concavity_mean = float(input("concavity (mean): "))
concave_points_mean = float(input("concave points (mean): "))
symmetry_mean = float(input("symmetry (mean): "))
fractal_dimension_mean = float(input("fractal dimension (mean): "))
radius_se = float(input("radius (standard error): "))
texture_se = float(input("texture (standard error): "))
perimeter_se = float(input("perimeter (standard error): "))
area_se = float(input("area (standard error): "))
smoothness_se = float(input("smoothness (standard error): "))
compactness_se = float(input("compactness (standard error): "))
concavity_se = float(input("concavity (standard error): "))
concave_points_se = float(input("concave points (standard error): "))
symmetry_se = float(input("symmetry (standard error): "))
fractal_dimension_se = float(input("fractal dimension (standard error): "))
radius_worst = float(input("radius (worst): "))
texture_worst = float(input("texture (worst): "))
perimeter_worst = float(input("perimeter (worst): "))
area_worst = float(input("area (worst): "))
smoothness_worst = float(input("smoothness (worst): "))
compactness_worst = float(input("compactness (worst): "))
concavity_worst = float(input("concavity (worst): "))
concave_points_worst = float(input("concave points (worst): "))
symmetry_worst = float(input("symmetry (worst): "))
fractal_dimension_worst = float(input("fractal dimension (worst): "))

data_set = np.array([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean,
                     concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se,
                     texture_se, perimeter_se, area_se, smoothness_se, compactness_se,
                     concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                     radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst,
                     concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst])

data_set = data_set[array_of_features]
ensemble_classifier = VotingClassifier(estimators=[('NN', MLPClassifier(alpha=1, max_iter=1000)),
                                                   ('AdaBoost', AdaBoostClassifier())
                                                   ,('Naive Baise', GaussianNB())
                                                   ,('Random Forest Classifer',
                                                     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
                                                   ], voting='soft')
classifiers_array = ["kNN", "Linear_SVM", "RBF_SVM", "Dec_Tree", "Rand_Forest",
                     "NN", "AdaBoost", "Naive Baise", "QDA", "Ensemble"]
classifiers = [
    KNeighborsClassifier(hyper_k),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth = hyper_depth, splitter="best"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    ensemble_classifier]
accuracy_array = []
prediction_array = []
i = 0
for classifier in classifiers:
    classifier.fit(data_X, data_y)
    y_pred = classifier.predict(unseen_test_data_X)
    probs = classifier.predict_proba(unseen_test_data_X)
    preds = probs[:,[1]]
    tn, fp, fn, tp = confusion_matrix(unseen_test_data_y, y_pred).ravel()
    accuracy_array.append(round(((tp+tn)/(tp+tn+fp+fn)), 2)*100)
    y = classifier.predict([data_set])
    for i in y:
        if i == 1:
            prediction_array.append("benign")
        elif i == 0:
            prediction_array.append("Malignant")

print('Patient (', name,') with the Emirates ID of: (',emirates_ID_number,'), is (',accuracy_array[-1],') % having brest cancer status of: (',prediction_array[-1],'). The following table illustrates each classifier prediction of her brest cancer status' )

tableArray = ["Classifier", "Accuracy (%)", "Prediction"]
data = [tableArray] + list(zip(classifiers_array, accuracy_array, prediction_array))
for i, d in enumerate(data):
    line = '|'.join(str(x).ljust(12) for x in d)
    print(line)
    if i == 0:
        print('-' * len(line))
