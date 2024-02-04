import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('Creditcard_data.csv')


X = df.drop('Class', axis=1)
y = df['Class']


ros = RandomOverSampler(sampling_strategy='auto')
X_resampled, y_resampled = ros.fit_resample(X, y)


results_data = pd.DataFrame(columns=['Model', 'Sampling', 'Accuracy'])


for i in range(1, 6):
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=i)

    
    models = [
        GaussianNB(),
        KNeighborsClassifier(),
        SVC(),
        LogisticRegression(solver='liblinear', max_iter=10000),
        DecisionTreeClassifier()
    ]

    
    samplings = [
        RandomOverSampler(),
        SMOTE(),
        SVMSMOTE(),
        RandomUnderSampler(),
        NearMiss()
    ]

    
    for model in models:
        model_name = model.__class__.__name__
        for sampler in samplings:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            model.fit(X_resampled, y_resampled)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            new_row = pd.DataFrame({'Model': [model_name], 'Sampling': [sampler.__class__.__name__], 'Accuracy': [accuracy]})
            results_data = pd.concat([results_data, new_row], ignore_index=True)


pivot_df = results_data.pivot_table(index='Model', columns='Sampling', values='Accuracy')


print(pivot_df)


pivot_df.to_csv('102103385_Sampling.csv')
