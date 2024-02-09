import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from functions import softmax, cross_entropy_loss
from training import train, predict

df = pd.read_csv('labels.csv')

class_val=[]
for i in range(len(df)):
    if df['Class'][i]=='no_crop':
        class_val.append(0)
    elif df['Class'][i]=='growing':
        class_val.append(1)
    else:
        class_val.append(2)

df['Class_val']=class_val
X=[]
y=[]

for i in range(len(df)):
    img_name=df['Image Label'][i]
    img=cv2.imread(f'prospace_assignment/{img_name}')
    resized_image=cv2.resize(img, (100, 100))
    X.append(resized_image)
    y.append(df['Class_val'][i])

X=np.array(X)
y=np.array(y)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
X_train=X_train / 255.0
X_test=X_test / 255.0
y_train=y_train.flatten()
y_test=y_test.flatten()

W, b = train(X_train, y_train)
y_pred = predict(X_test, W, b)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy}")
