
"""
重要知识点：
1. 逻辑回归模型中，模型参数w和w0的求解方法为最大似然估计，其优化目标函数为似然函数，优化方法为梯度下降法。
逻辑回归 原理简介：

Logistic回归虽然名字里带“回归”，但是它实际上是一种分类方法，主要用于两分类问题（即输出只有两种，分别代表两个类别），所以利用了Logistic函数（或称为Sigmoid函数），函数形式为：

logi(z)=1\(1+e^{-z})
逻辑回归从其原理上来说，逻辑回归其实是实现了一个决策边界：
对于函数 logi(z)=1\(1+e^{-z}) ,当  𝑧=>0时, 𝑦=>0.5,分类为1，
当  𝑧<0时, 𝑦<0.5,分类为0，其对应的 𝑦值我们可以视为类别1的概率预测值.

其对应的函数图像可以表示如下:
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.01)
y = 1/(1+np.exp(-x))

plt.plot(x,y)
plt.xlabel('z')
plt.ylabel('y')
plt.grid()
plt.show()

"""
step 1 库函数导入
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
step 2 数据读取
"""
#我们利用sklearn 中自带的iris数据作为数据载入，并利用pandas转换为DataFrame格式
from sklearn.datasets import load_iris
data = load_iris()#得到数据特征
iris_target = data.target#得到数据对应的标签
#利用Pandas转化为DataFrame格式
iris_df = pd.DataFrame(data.data, columns=data.feature_names)


"""
step 3 数据信息简单查看
"""
#数据类型和数据的形状
print(type(data.data))
print(data.data.shape)
print('------------------------------')
#利用.info()方法查看数据信息
iris_df.info()
print('------------------------------')
#进行简单的数据查看，我们可以利用.head（）头部 .tail（）尾部
print(iris_df.head())
print('------------------------------')
#其对应的类别标签，其中0，1，2分别代表'setosa', 'versicolor', 'virginica'三种不同花的类别
print(iris_target)
print('------------------------------')
#利用value_counts()查看标签的分布情况
print(pd.Series(iris_target).value_counts())
print('------------------------------')
#对特征进行一些统计描述
print(iris_df.describe())
print('------------------------------')

"""
step 4 数据可视化
"""
## 合并标签和特征信息
iris_all = iris_df.copy() ##进行浅拷贝，防止对于原始数据的修改
iris_all['target'] = iris_target
print(iris_all.head())
print(type(iris_all))
print('------------------------------')
## 特征与标签组合的散点可视化
sns.pairplot(data=iris_all,diag_kind='hist', hue= 'target')
plt.show()

for col in iris_df.columns:
    sns.boxplot(x='target', y=col, saturation=0.5,palette='pastel', data=iris_all)
    plt.title(col)
    plt.show()

#选取其前三个特征绘制三维散点图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,8))#设置画布大小
ax = fig.add_subplot(111, projection='3d')#添加子图

iris_all_class0 = iris_all[iris_all['target']==0].values
iris_all_class1 = iris_all[iris_all['target']==1].values
iris_all_class2 = iris_all[iris_all['target']==2].values
# 'setosa'(0), 'versicolor'(1), 'virginica'(2)
ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')
ax.scatter(iris_all_class1[:,0], iris_all_class1[:,1], iris_all_class1[:,2],label='versicolor')
ax.scatter(iris_all_class2[:,0], iris_all_class2[:,1], iris_all_class2[:,2],label='virginica')
plt.legend()
plt.show()
"""
step 5 利用逻辑回归模型在二分类上进行训练和预测
"""
## 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split

## 选择其类别为0和1的样本 （不包括类别为2的样本）
iris_features_part = iris_df.iloc[:100]
iris_target_part = iris_target[:100]

## 测试集大小为20%， 80%/20%分
x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size = 0.2, random_state = 2020)

#从sklearn中导入逻辑回归模型
from sklearn.linear_model import LogisticRegression
#定义逻辑回归模型
clf = LogisticRegression(random_state=0, solver='lbfgs')
#在训练集上训练逻辑回归模型
clf.fit(x_train, y_train)
## 查看其对应的w和对应的w0
print('the weight of Logistic Regression:',clf.coef_)
print('the intercept(w0) of Logistic Regression:',clf.intercept_)
print('------------------------------')
### 在训练集和测试集上分布获取预测结果
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
from sklearn import metrics
## 利用accuracy（准确率）【预测正确的样本数目占总样本数目的比例】作为评估指标
print('The accuracy of train set is {0}'.format(metrics.accuracy_score(y_train,train_predict)))
print('The accuracy of test set is {0}'.format(metrics.accuracy_score(y_test,test_predict)))
print('------------------------------')
#查看混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)
print('------------------------------')
#利用热力图显示混淆矩阵
plt.figure(figsize=(8,6))#设置画布大小
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')#绘制热力图
plt.xlabel('Predict label')#x轴标签
plt.ylabel('True label')#y轴标签
plt.show()

"""
step 6 利用逻辑回归模型在三分类上进行训练和预测
"""
## 测试集大小为20%， 80%/20%分
x_train, x_test, y_train, y_test = train_test_split(iris_df, iris_target, test_size = 0.2, random_state = 2020)
#定义逻辑回归模型
clf = LogisticRegression(random_state=0, solver='lbfgs')
#在训练集上训练逻辑回归模型
clf.fit(x_train, y_train)
## 查看其对应的w和对应的w0
print('the weight of Logistic Regression:\n',clf.coef_)
print('the intercept(w0) of Logistic Regression:\n',clf.intercept_)
print('------------------------------')
### 在训练集和测试集上分布获取预测结果
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
## 由于逻辑回归是概率预测模型，所有我们可以利用 predict_proba 函数预测其概率
train_predict_proba = clf.predict_proba(x_train)
test_predict_proba = clf.predict_proba(x_test)
print('The train predict Probability of each class:\n',train_predict_proba)
## 其中第一列代表预测为0类的概率，第二列代表预测为1类的概率，第三列代表预测为2类的概率。
print('------------------------------')
## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))
print('------------------------------')
# 查看混淆矩阵
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)
print('------------------------------')
# 利用热力图显示混淆矩阵
plt.figure(figsize=(8,6))#设置画布大小
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')#绘制热力图
plt.xlabel('Predict label')#x轴标签
plt.ylabel('True label')#y轴标签
plt.show()





