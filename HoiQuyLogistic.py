import numpy as np    
import pandas as pd  
import matplotlib.pyplot as plt            
from sklearn.linear_model import LogisticRegression 
duLieuTrain = pd.read_csv('train.csv')
duLieuTest = pd.read_csv('test.csv')
tuoiTrungBinh = duLieuTrain.groupby('Pclass').mean()['Age']
def hamTuoi(mang):
    tuoi = mang[0]
    Class = mang[1]
    
    if pd.isnull(tuoi):
        
        if  Class == 1:
            return tuoiTrungBinh[1]
        elif  Class == 2:
            return tuoiTrungBinh[2]
        else:
            return tuoiTrungBinh[3]
    else:
        return tuoi        
duLieuTrain['Age'] = duLieuTrain[['Age', 'Pclass']].apply(hamTuoi, axis=1)
duLieuTrain.drop('Cabin', axis=1, inplace=True)
duLieuTrain['Embarked'] = duLieuTrain['Embarked'].fillna(duLieuTrain['Embarked'].mode()[0])
tuoiTrungBinhTest = duLieuTest.groupby('Pclass').mean()['Age']
def hamTuoiTest(mang):
    tuoi = mang[0]
    Class = mang[1]
    
    if pd.isnull(tuoi):
        if  Class == 1:
            return tuoiTrungBinhTest[1]
        elif  Class == 2:
            return tuoiTrungBinhTest[2]
        else:
            return tuoiTrungBinhTest[3]
    else:
        return tuoi
duLieuTest['Age'] = duLieuTest[['Age', 'Pclass']].apply(hamTuoiTest, axis=1)
giaVeTest = duLieuTest.groupby('Pclass').mean()['Fare']
duLieuTest[duLieuTest['Fare'].isnull() == True]['Pclass']
duLieuTest['Fare'] = duLieuTest['Fare'].fillna(giaVeTest[3])
duLieuTest.drop('Cabin',axis=1, inplace=True)
gioiTinh = pd.get_dummies(duLieuTrain['Sex'], drop_first=True)
cangBien = pd.get_dummies(duLieuTrain['Embarked'], drop_first=True)
duLieuTrain.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis =1, inplace=True)
duLieuTrain = pd.concat([duLieuTrain, gioiTinh, cangBien], axis=1)
gioiTinh = pd.get_dummies(duLieuTest['Sex'], drop_first=True)
cangBien = pd.get_dummies(duLieuTest['Embarked'], drop_first=True)
duLieuTest.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis= 1, inplace=True)
duLieuTest = pd.concat([duLieuTest, gioiTinh, cangBien], axis =1)
Xtrain = duLieuTrain.drop(['Survived','PassengerId'], axis=1)
Ytrain = duLieuTrain['Survived']
Xtest = duLieuTest.drop('PassengerId', axis=1)
logmodel = LogisticRegression(max_iter=150)
logmodel.fit(Xtrain, Ytrain)
duDoan = logmodel.predict(Xtest)
ketqua = pd.DataFrame({
    'PassengerId': duLieuTest['PassengerId'],
    'Survived': duDoan
})

ketqua.to_csv('ketqua.csv', index = False)
