"""
step 1 库函数导入
"""
#基础函数库
import numpy as np

#导入画图库
import matplotlib.pyplot as plt
import seaborn as sns

#导入逻辑回归函数模型
from sklearn.linear_model import LogisticRegression

"""
step 2 模型训练
"""

#demo演示logisticregression分类

#构建数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])

#调用逻辑回归模型
lr_clf = LogisticRegression()

#用逻辑回归模型拟合构造的数据集
lr_clf = lr_clf.fit(x_fearures, y_label) #其拟合方程为y=w0+w1*x1+w2*x2

"""
step 3 模型参数查看
"""

#查看对应模型的w
print('the weight of Logistic Regression:',lr_clf.coef_)

#查看其他对应模型的w0
print('the intercept(w0) of Logistic Regression:',lr_clf.intercept_)

"""
step 4 数据和模型可视化
"""

#可视化构造的数据样本点
plt.figure()#创建一个图像窗口
plt.scatter(x_fearures[:,0],x_fearures[:,1],c=y_label,s=100,cmap='viridis')#散点图
plt.title('Logistic Regression')#图像标题
plt.show()#展示图像

#可视化决策边界
plt.figure()#创建一个图像窗口
plt.scatter(x_fearures[:,0],x_fearures[:,1],c=y_label,s=50,cmap='viridis')#散点图
plt.title('Logistic Regression')#图像标题

nx,ny = 200,100#生成网格点数
x_min,x_max = plt.xlim()#获取x轴最小值和最大值
y_min,y_max = plt.ylim()#获取y轴最小值和最大值
x_grid,y_grid = np.meshgrid(np.linspace(x_min,x_max,nx),np.linspace(y_min,y_max,ny))#生成网格采样点
z_proba = lr_clf.predict_proba(np.c_[x_grid.ravel(),y_grid.ravel()])#获取分类结果
z_proba = z_proba[:,1].reshape(x_grid.shape)#生成与网格点相对应的概率分类结果
plt.contour(x_grid,y_grid,z_proba,[0.5],linewidths=2.,colors='blue')#绘制分类边界线
plt.show()#展示图像

#可视化预测新样本
plt.figure()#创建一个图像窗口

#new point 1
x_fearures_new1 = np.array([[0,-1]])
plt.scatter(x_fearures_new1[:,0],x_fearures_new1[:,1],s=100,cmap='viridis')#散点图
plt.annotate('New Point 1',xy=(0,-1),xytext=(-2,0),color='red',arrowprops=dict(arrowstyle='->',color='red'))#注释

#new point 2
x_fearures_new2 = np.array([[1,2]])
plt.scatter(x_fearures_new2[:,0],x_fearures_new2[:,1],s=100,cmap='viridis')#散点图
plt.annotate('New Point 2',xy=(1,2),xytext=(-1.5,2.5),color='red',arrowprops=dict(arrowstyle='->',color='red'))#注释

#训练样本
plt.scatter(x_fearures[:,0],x_fearures[:,1],c=y_label,s=100,cmap='viridis')#散点图
plt.title('Logistic Regression')#图像标题

#可视化决策边界
plt.contour(x_grid,y_grid,z_proba,[0.5],linewidths=2.,colors='blue')#绘制分类边界线
plt.show()#展示图像

## 在训练集和测试集上分别利用训练好的模型进行预测
y_label_new1_predict = lr_clf.predict(x_fearures_new1)
y_label_new2_predict = lr_clf.predict(x_fearures_new2)

print('The New point 1 predict class:\n',y_label_new1_predict)
print('The New point 2 predict class:\n',y_label_new2_predict)

## 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所以我们可以利用 predict_proba 函数预测其概率
y_label_new1_predict_proba = lr_clf.predict_proba(x_fearures_new1)
y_label_new2_predict_proba = lr_clf.predict_proba(x_fearures_new2)

print('The New point 1 predict Probability of each class:\n',y_label_new1_predict_proba)
print('The New point 2 predict Probability of each class:\n',y_label_new2_predict_proba)



