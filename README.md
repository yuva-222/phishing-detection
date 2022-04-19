# phishing-detection
Phishing Website Detection by Machine Learning Techniques
1. Objective:
A phishing website is a common social engineering method that mimics trustful uniform resource locators (URLs) and webpages. The objective of this project is to train machine learning models and deep neural nets on the dataset created to predict phishing websites. Both phishing and benign URLs of websites are gathered to form a dataset and from them required URL and website content-based features are extracted. The performance level of each model is measures and compared.

This project is worked on Google Collaboratory. The required packages for this notebook are imported when needed.

2. Loading Data:
The features are extracted and store in the csv file. The working of this can be seen in the 'Phishing Website Detection_Feature Extraction.ipynb' file.

The reulted csv file is uploaded to this notebook and stored in the dataframe.

#importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Loading the data
data0 = pd.read_csv("C:/Users/yuvak/OneDrive/Desktop/Phising_Training_Dataset1.csv")
data0.head()
key	having_IP	URL_Length	Shortining_Service	having_At_Symbol	double_slash_redirecting	Prefix_Suffix	having_Sub_Domain	SSLfinal_State	Domain_registeration_length	...	popUpWidnow	Iframe	age_of_domain	DNSRecord	web_traffic	Page_Rank	Google_Index	Links_pointing_to_page	Statistical_report	Result
0	12344	-1	1	1	1	-1	-1	-1	-1	-1	...	1	1	-1	-1	-1	-1	1	1	-1	-1
1	12345	1	1	1	1	1	-1	0	1	-1	...	1	1	-1	-1	0	-1	1	1	1	-1
2	12346	1	0	1	1	1	-1	-1	-1	-1	...	1	1	1	-1	1	-1	1	0	-1	-1
3	12347	1	0	1	1	1	-1	-1	-1	1	...	1	1	-1	-1	1	-1	1	-1	1	-1
4	12348	1	0	-1	1	1	-1	1	1	-1	...	-1	1	-1	-1	0	-1	1	1	1	1
5 rows × 32 columns

3. Familiarizing with Data
In this step, few dataframe methods are used to look into the data and its features.

#Checking the shape of the dataset
data0.shape
(8955, 32)
#Listing the features of the dataset
data0.columns
Index(['key', 'having_IP', 'URL_Length', 'Shortining_Service',
       'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
       'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
       'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
       'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
       'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
       'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
       'Google_Index', 'Links_pointing_to_page', 'Statistical_report',
       'Result'],
      dtype='object')
#Information about the dataset
data0.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8955 entries, 0 to 8954
Data columns (total 32 columns):
 #   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
 0   key                          8955 non-null   int64
 1   having_IP                    8955 non-null   int64
 2   URL_Length                   8955 non-null   int64
 3   Shortining_Service           8955 non-null   int64
 4   having_At_Symbol             8955 non-null   int64
 5   double_slash_redirecting     8955 non-null   int64
 6   Prefix_Suffix                8955 non-null   int64
 7   having_Sub_Domain            8955 non-null   int64
 8   SSLfinal_State               8955 non-null   int64
 9   Domain_registeration_length  8955 non-null   int64
 10  Favicon                      8955 non-null   int64
 11  port                         8955 non-null   int64
 12  HTTPS_token                  8955 non-null   int64
 13  Request_URL                  8955 non-null   int64
 14  URL_of_Anchor                8955 non-null   int64
 15  Links_in_tags                8955 non-null   int64
 16  SFH                          8955 non-null   int64
 17  Submitting_to_email          8955 non-null   int64
 18  Abnormal_URL                 8955 non-null   int64
 19  Redirect                     8955 non-null   int64
 20  on_mouseover                 8955 non-null   int64
 21  RightClick                   8955 non-null   int64
 22  popUpWidnow                  8955 non-null   int64
 23  Iframe                       8955 non-null   int64
 24  age_of_domain                8955 non-null   int64
 25  DNSRecord                    8955 non-null   int64
 26  web_traffic                  8955 non-null   int64
 27  Page_Rank                    8955 non-null   int64
 28  Google_Index                 8955 non-null   int64
 29  Links_pointing_to_page       8955 non-null   int64
 30  Statistical_report           8955 non-null   int64
 31  Result                       8955 non-null   int64
dtypes: int64(32)
memory usage: 2.2 MB
4. Visualizing the data
Few plots and graphs are displayed to find how the data is distributed and the how features are related to each other.

#Plotting the data distribution
data0.hist(bins = 50,figsize = (15,15))
plt.show()

#Correlation heatmap
​
plt.figure(figsize=(15,13))
sns.heatmap(data0.corr())
plt.show()

5. Data Preprocessing & EDA
Here, we clean the data by applying data preprocesssing techniques and transform the data to use it in the models.

data0.describe()
key	having_IP	URL_Length	Shortining_Service	having_At_Symbol	double_slash_redirecting	Prefix_Suffix	having_Sub_Domain	SSLfinal_State	Domain_registeration_length	...	popUpWidnow	Iframe	age_of_domain	DNSRecord	web_traffic	Page_Rank	Google_Index	Links_pointing_to_page	Statistical_report	Result
count	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.00000	...	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000	8955.000000
mean	16821.000000	0.307203	-0.635734	0.740480	0.709436	0.740704	-0.735343	0.071803	0.264545	-0.33646	...	0.606700	0.829816	0.028922	0.371078	0.291792	-0.479397	0.712339	0.338582	0.728867	0.124288
std	2585.230164	0.951697	0.763660	0.672116	0.704809	0.671870	0.677733	0.817419	0.908003	0.94175	...	0.794975	0.558069	0.999637	0.928654	0.825557	0.877647	0.701874	0.576068	0.684694	0.992302
min	12344.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.00000	...	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000
25%	14582.500000	-1.000000	-1.000000	1.000000	1.000000	1.000000	-1.000000	-1.000000	-1.000000	-1.00000	...	1.000000	1.000000	-1.000000	-1.000000	0.000000	-1.000000	1.000000	0.000000	1.000000	-1.000000
50%	16821.000000	1.000000	-1.000000	1.000000	1.000000	1.000000	-1.000000	0.000000	1.000000	-1.00000	...	1.000000	1.000000	1.000000	1.000000	1.000000	-1.000000	1.000000	0.000000	1.000000	1.000000
75%	19059.500000	1.000000	-1.000000	1.000000	1.000000	1.000000	-1.000000	1.000000	1.000000	1.00000	...	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000
max	21298.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.00000	...	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000
8 rows × 32 columns

#Dropping the Domain column
data = data0.drop([], axis = 1).copy()
#checking the data for null or missing values
data.isnull().sum()
key                            0
having_IP                      0
URL_Length                     0
Shortining_Service             0
having_At_Symbol               0
double_slash_redirecting       0
Prefix_Suffix                  0
having_Sub_Domain              0
SSLfinal_State                 0
Domain_registeration_length    0
Favicon                        0
port                           0
HTTPS_token                    0
Request_URL                    0
URL_of_Anchor                  0
Links_in_tags                  0
SFH                            0
Submitting_to_email            0
Abnormal_URL                   0
Redirect                       0
on_mouseover                   0
RightClick                     0
popUpWidnow                    0
Iframe                         0
age_of_domain                  0
DNSRecord                      0
web_traffic                    0
Page_Rank                      0
Google_Index                   0
Links_pointing_to_page         0
Statistical_report             0
Result                         0
dtype: int64
In the feature extraction file, the extracted features of legitmate & phishing url datasets are just concatenated without any shuffling. This resulted in top 5000 rows of legitimate url data & bottom 5000 of phishing url data.

To even out the distribution while splitting the data into training & testing sets, we need to shuffle it. This even evades the case of overfitting while model training.

# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data.sample(frac=1).reset_index(drop=True)
data.head()
key	having_IP	URL_Length	Shortining_Service	having_At_Symbol	double_slash_redirecting	Prefix_Suffix	having_Sub_Domain	SSLfinal_State	Domain_registeration_length	...	popUpWidnow	Iframe	age_of_domain	DNSRecord	web_traffic	Page_Rank	Google_Index	Links_pointing_to_page	Statistical_report	Result
0	13641	1	-1	1	1	1	-1	-1	-1	-1	...	-1	1	-1	-1	1	-1	1	1	1	-1
1	16716	-1	-1	1	1	1	-1	0	0	1	...	1	1	-1	1	0	-1	1	1	-1	-1
2	18828	1	0	1	1	1	-1	0	-1	1	...	1	1	-1	1	0	1	1	0	1	-1
3	17975	-1	-1	1	1	1	-1	0	0	1	...	1	1	-1	1	-1	-1	1	0	-1	-1
4	16234	1	-1	1	1	1	-1	-1	-1	-1	...	1	1	-1	1	0	-1	1	0	1	-1
5 rows × 32 columns

From the above execution, it is clear that the data doesnot have any missing values.

By this, the data is throughly preprocessed & is ready for training.

6. Splitting the Data
ult
# Sepratating & assigning features and target columns to X & y
y = data['Result']
X = data.drop('Result',axis=1)
X.shape, y.shape
((8955, 31), (8955,))
# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split
​
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape
((7164, 31), (1791, 31))
7. Machine Learning Models & Training
From the dataset above, it is clear that this is a supervised machine learning task. There are two major types of supervised machine learning problems, called classification and regression.

This data set comes under classification problem, as the input URL is classified as phishing (1) or legitimate (0). The supervised machine learning models (classification) considered to train the dataset in this notebook are:

Decision Tree Random Forest Multilayer Perceptrons XGBoost Autoencoder Neural Network Support Vector Machines

#importing packages
from sklearn.metrics import accuracy_score
# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []
​
#function to call for storing the results
def storeResults(model, a,b):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))
7.1. Decision Tree Classifier
Decision trees are widely used models for classification and regression tasks. Essentially, they learn a hierarchy of if/else questions, leading to a decision. Learning a decision tree means learning the sequence of if/else questions that gets us to the true answer most quickly.

In the machine learning setting, these questions are called tests (not to be confused with the test set, which is the data we use to test to see how generalizable our model is). To build a tree, the algorithm searches over all possible tests and finds the one that is most informative about the target variable.

# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier
​
# instantiate the model 
tree = DecisionTreeClassifier(max_depth = 5)
# fit the model 
tree.fit(X_train, y_train)
DecisionTreeClassifier(max_depth=5)
#predicting the target value from the model for the samples
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)
Performance Evaluation:

)
#computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train,y_train_tree)
acc_test_tree = accuracy_score(y_test,y_test_tree)
​
print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))
Decision Tree: Accuracy on training Data: 0.929
Decision Tree: Accuracy on test Data: 0.916
#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()

Storing the results:

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Decision Tree', acc_train_tree, acc_test_tree)
7.2. Random Forest Classifier
Random forests for regression and classification are currently among the most widely used machine learning methods.A random forest is essentially a collection of decision trees, where each tree is slightly different from the others. The idea behind random forests is that each tree might do a relatively good job of predicting, but will likely overfit on part of the data.

If we build many trees, all of which work well and overfit in different ways, we can reduce the amount of overfitting by averaging their results. To build a random forest model, you need to decide on the number of trees to build (the n_estimators parameter of RandomForestRegressor or RandomForestClassifier). They are very powerful, often work well without heavy tuning of the parameters, and don’t require scaling of the data.

# Random Forest model
from sklearn.ensemble import RandomForestClassifier
​
# instantiate the model
forest = RandomForestClassifier(max_depth=5)
​
# fit the model 
forest.fit(X_train, y_train)
RandomForestClassifier(max_depth=5)
#predicting the target value from the model for the samples
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)
Performance Evaluation:

#computing the accuracy of the model performance
acc_train_forest = accuracy_score(y_train,y_train_forest)
acc_test_forest = accuracy_score(y_test,y_test_forest)
​
print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))
Random forest: Accuracy on training Data: 0.934
Random forest: Accuracy on test Data: 0.920
#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()

Storing the results:

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Random Forest', acc_train_forest, acc_test_forest)
7.3. Multilayer Perceptrons (MLPs): Deep Learning
Multilayer perceptrons (MLPs) are also known as (vanilla) feed-forward neural networks, or sometimes just neural networks. Multilayer perceptrons can be applied for both classification and regression problems.

MLPs can be viewed as generalizations of linear models that perform multiple stages of processing to come to a decision.

# Multilayer Perceptrons model
from sklearn.neural_network import MLPClassifier
​
# instantiate the model
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))
​
# fit the model 
mlp.fit(X_train, y_train)
MLPClassifier(alpha=0.001, hidden_layer_sizes=[100, 100, 100])
#predicting the target value from the model for the samples
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)
Performance Evaluation:

#computing the accuracy of the model performance
acc_train_mlp = accuracy_score(y_train,y_train_mlp)
acc_test_mlp = accuracy_score(y_test,y_test_mlp)
​
print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))
Multilayer Perceptrons: Accuracy on training Data: 0.625
Multilayer Perceptrons: Accuracy on test Data: 0.622
Storing the results:

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)
7.4. Support Vector Machines
In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.

#Support vector machine model
from sklearn.svm import SVC
​
# instantiate the model
svm = SVC(kernel='linear', C=1.0, random_state=12)
#fit the model
svm.fit(X_train, y_train)
SVC(kernel='linear', random_state=12)
#predicting the target value from the model for the samples
y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)
Performance Evaluation:

#computing the accuracy of the model performance
acc_train_svm = accuracy_score(y_train,y_train_svm)
acc_test_svm = accuracy_score(y_test,y_test_svm)
​
print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))
SVM: Accuracy on training Data: 0.919
SVM : Accuracy on test Data: 0.906
Storing the results:

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('SVM', acc_train_svm, acc_test_svm)
7.5. Autoencoder Neural Network
An auto encoder is a neural network that has the same number of input neurons as it does outputs. The hidden layers of the neural network will have fewer neurons than the input/output neurons. Because there are fewer neurons, the auto-encoder must learn to encode the input to the fewer hidden neurons. The predictors (x) and output (y) are exactly the same in an auto encoder.

pip install tensor flow
Requirement already satisfied: tensor in c:\users\yuvak\anaconda3\lib\site-packages (0.3.6)
Requirement already satisfied: flow in c:\users\yuvak\anaconda3\lib\site-packages (0.0.1)
Requirement already satisfied: PyYaml in c:\users\yuvak\anaconda3\lib\site-packages (from tensor) (5.3.1)
Requirement already satisfied: protobuf in c:\users\yuvak\anaconda3\lib\site-packages (from tensor) (3.20.0)
Requirement already satisfied: construct in c:\users\yuvak\anaconda3\lib\site-packages (from tensor) (2.10.68)
Requirement already satisfied: Twisted in c:\users\yuvak\anaconda3\lib\site-packages (from tensor) (22.4.0)
Requirement already satisfied: pysnmp in c:\users\yuvak\anaconda3\lib\site-packages (from tensor) (4.4.12)
Requirement already satisfied: Automat>=0.8.0 in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (20.2.0)
Requirement already satisfied: zope.interface>=4.4.2 in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (5.1.2)
Requirement already satisfied: constantly>=15.1 in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (15.1.0)
Requirement already satisfied: attrs>=19.2.0 in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (20.3.0)
Requirement already satisfied: typing-extensions>=3.6.5 in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (3.7.4.3)
Requirement already satisfied: hyperlink>=17.1.1 in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (21.0.0)
Requirement already satisfied: incremental>=21.3.0 in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (21.3.0)
Requirement already satisfied: twisted-iocpsupport<2,>=1.0.2; platform_system == "Windows" in c:\users\yuvak\anaconda3\lib\site-packages (from Twisted->tensor) (1.0.2)
Requirement already satisfied: pysmi in c:\users\yuvak\anaconda3\lib\site-packages (from pysnmp->tensor) (0.3.4)
Requirement already satisfied: pycryptodomex in c:\users\yuvak\anaconda3\lib\site-packages (from pysnmp->tensor) (3.14.1)
Requirement already satisfied: pyasn1>=0.2.3 in c:\users\yuvak\anaconda3\lib\site-packages (from pysnmp->tensor) (0.4.8)
Requirement already satisfied: six in c:\users\yuvak\anaconda3\lib\site-packages (from Automat>=0.8.0->Twisted->tensor) (1.15.0)
Requirement already satisfied: setuptools in c:\users\yuvak\anaconda3\lib\site-packages (from zope.interface>=4.4.2->Twisted->tensor) (50.3.1.post20201107)
Requirement already satisfied: idna>=2.5 in c:\users\yuvak\anaconda3\lib\site-packages (from hyperlink>=17.1.1->Twisted->tensor) (2.10)
Requirement already satisfied: ply in c:\users\yuvak\anaconda3\lib\site-packages (from pysmi->pysnmp->tensor) (3.11)
Note: you may need to restart the kernel to use updated packages.
pip install keras
pip install keras
Requirement already satisfied: keras in c:\users\yuvak\anaconda3\lib\site-packages (2.8.0)
Note: you may need to restart the kernel to use updated packages.
w
pip show tensorflow
Note: you may need to restart the kernel to use updated packages.
WARNING: Package(s) not found: tensorflow
#importing required packages
import keras
from keras.layers import Input, Dense
from keras import regularizers
import tensorflow as tf
from keras.models import Model
from sklearn import metrics
#building autoencoder model
​
input_dim = X_train.shape[1]
encoding_dim = input_dim
​
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)
​
encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim-2), activation='relu')(code)
​
decoder = Dense(int(encoding_dim), activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 31)]              0         
                                                                 
 dense (Dense)               (None, 31)                992       
                                                                 
 dense_1 (Dense)             (None, 31)                992       
                                                                 
 dense_2 (Dense)             (None, 29)                928       
                                                                 
 dense_5 (Dense)             (None, 31)                930       
                                                                 
 dense_6 (Dense)             (None, 31)                992       
                                                                 
=================================================================
Total params: 4,834
Trainable params: 4,834
Non-trainable params: 0
_________________________________________________________________
#compiling the model
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
​
#Training the model
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2) 
Epoch 1/10
90/90 [==============================] - 1s 5ms/step - loss: -8226.4834 - accuracy: 0.0000e+00 - val_loss: -8240.6074 - val_accuracy: 0.0000e+00
Epoch 2/10
90/90 [==============================] - 0s 3ms/step - loss: -8229.6914 - accuracy: 0.0000e+00 - val_loss: -8242.7422 - val_accuracy: 0.0000e+00
Epoch 3/10
90/90 [==============================] - 0s 3ms/step - loss: -8232.1035 - accuracy: 0.0000e+00 - val_loss: -8245.3271 - val_accuracy: 0.0000e+00
Epoch 4/10
90/90 [==============================] - 0s 3ms/step - loss: -8231.7520 - accuracy: 0.1178 - val_loss: -8246.1338 - val_accuracy: 1.0000
Epoch 5/10
90/90 [==============================] - 0s 2ms/step - loss: -8234.2295 - accuracy: 1.0000 - val_loss: -8246.1484 - val_accuracy: 1.0000
Epoch 6/10
90/90 [==============================] - 0s 2ms/step - loss: -8234.2705 - accuracy: 1.0000 - val_loss: -8246.1973 - val_accuracy: 1.0000
Epoch 7/10
90/90 [==============================] - 0s 3ms/step - loss: -8234.3252 - accuracy: 1.0000 - val_loss: -8246.2510 - val_accuracy: 1.0000
Epoch 8/10
90/90 [==============================] - 0s 3ms/step - loss: -8234.3818 - accuracy: 1.0000 - val_loss: -8246.3105 - val_accuracy: 1.0000
Epoch 9/10
90/90 [==============================] - 0s 3ms/step - loss: -8234.4404 - accuracy: 1.0000 - val_loss: -8246.3750 - val_accuracy: 1.0000
Epoch 10/10
90/90 [==============================] - 0s 2ms/step - loss: -8234.5068 - accuracy: 1.0000 - val_loss: -8246.4463 - val_accuracy: 1.0000
Performance Evaluation:

acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]
​
print('\nAutoencoder: Accuracy on training Data: {:.3f}' .format(acc_train_auto))
print('Autoencoder: Accuracy on test Data: {:.3f}' .format(acc_test_auto))
224/224 [==============================] - 0s 662us/step - loss: -8236.9238 - accuracy: 1.0000
56/56 [==============================] - 0s 785us/step - loss: -8266.8125 - accuracy: 1.0000

Autoencoder: Accuracy on training Data: 1.000
Autoencoder: Accuracy on test Data: 1.000
Storing the results:

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('AutoEncoder', acc_train_auto, acc_test_auto)
8. Comparision of Models
To compare the models performance, a dataframe is created. The columns of this dataframe are the lists created to store the results of the model.

#creating dataframe
results = pd.DataFrame({ 'ML Model': ML_Model,    
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
results
ML Model	Train Accuracy	Test Accuracy
0	Decision Tree	0.929	0.916
1	Random Forest	0.934	0.920
2	Multilayer Perceptrons	0.625	0.622
3	SVM	0.919	0.906
4	AutoEncoder	1.000	1.000
#Sorting the datafram on accuracy
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)
ML Model	Train Accuracy	Test Accuracy
4	AutoEncoder	1.000	1.000
1	Random Forest	0.934	0.920
0	Decision Tree	0.929	0.916
3	SVM	0.919	0.906
2	Multilayer Perceptrons	0.625	0.622
9. References
https://blog.keras.io/building-autoencoders-in-keras.html

https://en.wikipedia.org/wiki/Autoencoder

https://machinelearningmastery.com/save-gradient-boosting-models-xgboost-python/

​
