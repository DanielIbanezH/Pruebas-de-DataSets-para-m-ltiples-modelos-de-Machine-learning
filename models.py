# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings


def multiclass_specificity(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    num_classes = conf_matrix.shape[0]
    specificity_values = []
    for i in range(num_classes):
        # Calcula la especificidad para la clase i
        tn = np.sum(conf_matrix[i, j] for j in range(num_classes) if j != i)
        fp = np.sum(conf_matrix[j, i] for j in range(num_classes) if j != i)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Evita divisiones por cero
        specificity_values.append(specificity)
    # Calcula la especificidad promediada
    mean_specificity = np.mean(specificity_values)
    return mean_specificity

def modelDataSet(dataSetName):
    if dataSetName == "AutoInsurSweden":
        data = pd.read_csv('AutoInsurSweden.csv')
        umbral_pago_alto = 200  # Por ejemplo, en miles de coronas suecas
        # Dividir los datos en características (X) y etiquetas (y)
        data['etiqueta'] = np.where(data.iloc[:, 1] > umbral_pago_alto, 1, 0)
        X = data.iloc[:, :-1].values
        y = data['etiqueta'].values
    elif dataSetName == "pima-indians-diabetes":
        data = pd.read_csv('pima-indians-diabetes.csv')
        X=data.iloc[:, :-1]
        y=data.iloc[:, -1]
    elif dataSetName == "winequality-white":
        data = pd.read_csv('winequality-white.csv', sep=';')
        X=data.iloc[:, :-1]
        y=data.iloc[:, -1]
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalizar los datos (solo para algoritmos sensibles a la escala como KNN y SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Inicializar los modelos
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machines': SVC(),
        'Naive Bayes': GaussianNB(),
    }
    # Entrenar y evaluar los modelos
    # Dentro de la función modelDataSet justo antes de entrar al bucle for
    warnings.filterwarnings("ignore")
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled if 'K-Nearest Neighbors' in name or 'Support Vector Machines' in name or "winequality-white" in dataSetName else X_train, y_train)
        y_pred = model.predict(X_test_scaled if 'K-Nearest Neighbors' in name or 'Support Vector Machines' in name or "winequality-white" in dataSetName else X_test)
        if dataSetName=="winequality-white":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) 
            recall = recall_score(y_test, y_pred, average='weighted',zero_division=1)
            f1 = f1_score(y_test, y_pred,  average='weighted')
            specificity = multiclass_specificity(y_test, y_pred)  # Calcula la especificidad promediada
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred) 
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': specificity
        }
        
    print(f'================{dataSetName}================')
    for name, result in results.items():
        print(f"Results for {name}:")
        for metric, value in result.items():
            print(f"{metric}: {value}")
        print("\n")




modelDataSet("AutoInsurSweden")
modelDataSet("pima-indians-diabetes")
modelDataSet("winequality-white")
    

