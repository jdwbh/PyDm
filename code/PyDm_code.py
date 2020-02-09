## -*- coding: utf-8 -*-
"""
###   Python数据挖掘方法及应用（PyDm）
###   函数库 PyDm_code.py
###   数据框 PyDm_data.xlsx
###   王斌会 王术 2018-6-2
"""

#系统初始化
run initpy.py
from PyDm_fun import *
### 读数据csv（本地）
#BSdata=pd.read_csv('BSdata.csv',encoding='utf-8'); BSdata[:6]
### 读数据csv（云端）
#url1='http://leanote.com/api/file/getAttach?fileId=5abbb388ab6441507e002161'
#dat1=pd.read_csv(url1,encoding='utf-8'); dat1

### 读数据xlsx（本地）
BSdata=pd.read_excel('PyDm_data.xlsx','BSdata'); BSdata[:6]
### 读数据csv（云端）
#url2='http://leanote.com/api/file/getAttach?fileId=5abbb3aaab6441507e002167'
#dat2=pd.read_excel(url2,'BSdata'); dat2

#
#第1章 数据收集与软件应用
##1.3 Python编程基础
#### 1.3.1.1 Python的工作目录
'''获得当前目录'''
pwd
'''改变工作目录'''
cd "D:\\PyDm"
pwd

!dir

###1.3.2 Python基本数据类型
####1.3.2.1 数据对象及类型
#列举当前环境中的对象名
who
x=10   #创建对象x
whos
del x   #删除对象x
who_ls

####1.3.2.2 数据基本类型
#数字
n=10       #整数
n
print("n=",n)
x=10.234   #实数
x
print("x=%10.5f"%x)

a=True;a
b=False;b

10>3
10<3

# 字符串
s = 'I love Python';s
s[7]
s[2:6]
s+s
s*2

float('nan')

####1.3.3.3 标准数据类型
#（1）List（列表）
list1 =[] # 空列表
list1
list1 = ['Python', 786 , 2.23, 'R', 70.2]
list1 # 输出完整列表
list1[0] # 输出列表的第一个元素
list1[1:3] # 输出第二个至第三个元素
list1[2:] # 输出从第三个开始至列表末尾的所有元素
list1 * 2 # 输出列表两次
list1 + list1[2:4] # 打印组合的列表

X=[1,3,6,4,9]; X
sex=['女','男','男','女','男']
sex
weight=[67,66,83,68,70];
weight

#（2）Tuple（元组）

#（3）Dictionary（字典）
#字典
{}            #空字典
dict1={'name':'john','code':6734,'dept':'sales'};dict1 #定义字典
dict1['code']  # 输出键为'code' 的值
dict1.keys()   # 输出所有键
dict1.values() # 输出所有值

dict2={'sex': sex,'weight':weight}; dict2 #根据列表构成字典

###1.3.3 数值分析库numpy
####1.3.3.1 一维数组
import numpy as np       #加载数组包
np.array([1,2,3,4,5])         #一维数组
np.array([1,2,3,np.nan,5])    #包含缺失值的数组

np.arange(9)             #数组序列
np.arange(1,9,0.5)       #等差数列
np.linspace(1,9,5)       #等距数列

np.random.randint(1,9)   #1~9随机数
np.random.rand(10)       #10个均匀随机数
np.random.randn(10)      #10个正态随机数

####1.3.3.2 二维数组
np.array([[1,2],[3,4],[5,6]])   #二维数组

####1.3.3.3. 数组的基本操作
A=np.arange(9).reshape(3,3);A        #形成矩阵
A.shape
np.empty([3,3]) #空数组
np.zeros((3,3)) #零矩阵
np.ones((3,3))  #1矩阵
np.eye(3) #单位阵

###1.3.4 数据分析库pandas
import pandas as pd   #加载数据分析包

####1.3.4.1 序列: Seriers
#（1）创建序列（向量、一维数组）
pd.Series()           #生成空序列
X=[1,3,6,4,9]
S1=pd.Series(X);S1
S2=pd.Series(weight);S2
S3=pd.Series(sex);S3

pd.concat([S2,S3],axis=0)    #按行并序列
pd.concat([S2,S3],axis=1)    #按列并序列

S1[2]
S3[1:4]

####1.3.4.2 数据框: DataFrame
#（1）根据列表创建数据框
pd.DataFrame()      #生成空数据框
pd.DataFrame(X, columns=['X'], index=range(5))
pd.DataFrame(weight,columns=['weight'], index=['A','B','C','D','E'])

#（2）根据字典创建数据框
'''通过字典列表生成数据框是Python较快捷的方式 '''
df1=pd.DataFrame({'S1':S1,'S2':S2,'S3':S3});df1
df2=pd.DataFrame({'sex':sex,'weight':weight},index=X);df2

#（3）增加数据框列
df2['weight2']=df2.weight**2; df2   # 生成新列

#（4）删除数据框列
del df2['weight2']; df2   #删除数据列

df3=pd.DataFrame({'S2':S2,'S3':S3},index=S1);df3

df3.isnull()#是缺失值返回True，否则范围False
df3.isnull().sum()#返回每列包含的缺失值的个数
df3.dropna()   #直接删除含有缺失值的行，多变量谨慎使用
#df3.dropna(how = 'all')#只删除全是缺失值的行
#（5）数据框排序
df3.sort_index()         #按index排序
df3.sort_values(by='S3') #按列值排序

####1.3.4.3 数据框的读和写
#（1）pandas读取数据集
#BSdata=pd.read_clipboard();BSdata[:5]  #从剪切板上复制数据
BSdata=pd.read_csv("BSdata.csv",encoding='utf-8');BSdata[6:9]
BSdata=pd.read_excel('PyDm_data.xlsx','BSdata');BSdata[-5:]

#（2）pandas数据集的保存
BSdata.to_csv('BSdata1.csv') # 将数据框BSdata存 保存到 到BSdata.csv

####1.3.4.4 数据框的操作
#（1）获取数据框的基本信息
BSdata.info()            #数据框信息

BSdata.head()            #显示前5行
BSdata.tail()            #显示后5行
BSdata.columns           #查看列名称

BSdata.index             #数据框行名
BSdata.values            #数据框值数组
BSdata.shape             #显示数据框的行数和列数
BSdata.shape[0]          #数据框行数
BSdata.shape[1]          #数据框列数

#（2）选取变量
BSdata.性别              #取一列数据，BSdata['性别']
BSdata[['身高','体重']]  #取两列数据
BSdata[:3]               #BSdata.head(3)
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5]       #0到2行和1:5列数据

#（3）选取观测与变量
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5]       #0到2行和1:5列数据

#（4）根据条件选取样品与变量
BSdata[BSdata['身高']>180]

# (5) query法
BSdata.query('身高>180')
BSdata.query('身高>180 and 体重>80')

#（6）数据框的运算
BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],2)
del BSdata['体重指数']   #删除数据列

pd.concat([BSdata.身高, BSdata.体重],axis=0)
pd.concat([BSdata.身高, BSdata.体重],axis=1)

BSdata.iloc[:3,:5].T

###1.3.3 Python编程运算
####1.3.3.1 基本运算
####1.3.3.2 控制语句
#（1）循环语句for
for i in range(1,5):
     print(i)

fruits = ['banana', 'apple',  'mango']
for fruit in fruits:
   print('当前水果 :', fruit)

for num in range(10,15):
    print(num)

for var in BSdata.columns:
    print(var)

#（2）条件语句if/else
a = -100
if a < 100:
    print("数值小于100")
else:
    print("数值大于100")

-a if a<0 else a

#### 1.3.3.3 函数定义
#### 1.3.3.4 面向对象
x=[1,3,6,4,9,7,5,8,2]; x
def xbar(x):
    n=len(x)
    xm=sum(x)/n
    return(xm)

xbar(x)

np.mean(x)

X=np.array([1,3,6,4,9,7,5,8,2]);X # 列表数组
def SS1(x):
    n=len(x)
    ss=sum(x**2)-sum(x)**2/n
    return(ss)
SS1(X) #SS1(BSdata. 身高)

def SS2(x): # 返回多个值
    n=len(x)
    xm=sum(x)/n
    ss=sum(x**2)-sum(x)**2/n
    return[x**2,n,xm,ss] #return(x**2,n,xm,ss)
SS2(X) #SS2(BSdata. 身高)

SS2(X)[0] # 取第1 个对象
SS2(X)[1] # 取第2 个对象
SS2(X)[2] # 取第3 个对象
SS2(X)[3] # 取第4 个对象

type(SS2(X))
type(SS2(X)[3])
#数据及练习1

# 第2章 探索性数据分析
## 2.1 数据的描述统计
### 2.1.1 基本统计量
BSdata.describe()
BSdata[['性别','开设','课程','软件']].describe()

#### 2.1.1.1 计数数据的汇总分析
#（1）频数：绝对数
T1=BSdata.性别.value_counts();T1

#（2）频率：相对数
T1/sum(T1)*100

####2.1.1.2 计量数据的汇总分析	41
#（1）均数（算术平均数）
BSdata.身高.mean()
#（2）中位数
BSdata.身高.median()
#（3）极差
BSdata.身高.max()-BSdata.身高.min()
#（4）方差
BSdata.身高.var()
#（5）标准差
BSdata.身高.std()
#（6）四分位数间距
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)
#（7）偏度
BSdata.身高.skew()
#（8）峰度
BSdata.身高.kurt()

#（9）自定义计算基本统计量函数
def stats(x):
    stat=[x.count(),x.min(),x.quantile(.25),x.mean(),x.median(),
         x.quantile(.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min', 'Q1(25%)','Mean','Median',
                   'Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    return(stat)
stats(BSdata.身高)

#加载自定义函数库
import PyDm1func as da
da.stats(BSdata.身高)
da.stats(BSdata.支出)

###2.1.2 基本统计图
####2.1.2.1 matlibplot绘图函数
import matplotlib.pyplot as plt              #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi'];   #SimHei黑体
plt.rcParams['axes.unicode_minus']=False;    #正常显示图中负号
plt.figure(figsize=(5,4));                   #图形大小
'''本地直接显示图形'''
%matplotlib inline

#（1）常用的统计图函数
#（2）图形参数设置
####二、计数数据的基本统计图
X=['A','B','C','D','E','F','G']
Y=[1,4,7,3,2,5,6]
plt.bar(X,Y); # 条图
plt.pie(Y,labels=X);  # 饼图

####三、计量数据的基本统计图
plt.plot(X,Y);  #线图 plot

plt.hist(BSdata.身高)  # 频数直方图
plt.hist(BSdata.身高,density=True) # 频率直方图

plt.scatter(BSdata.身高, BSdata.体重);  # 散点图
plt.xlabel(u'身高');plt.ylabel(u'体重');

#（3）图形参数设置
plt.plot(X,Y,c='red');
plt.ylim(0,8);
plt.xlabel('names');plt.ylabel('values');
plt.xticks(range(len(X)), X);
plt.plot(X,Y,linestyle='--',marker='o');

plt.plot(X,Y,'o--'); plt.axvline(x=1);plt.axhline(y=4);
#plt.vlines(1,0,6,colors='r');plt.hlines(4,0,6);

plt.plot(X,Y,label=u'折线');
plt.legend();

#误差条图
s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5})

#（4）多图
plt.figure(figsize=(5,4));
plt.subplot(121); plt.bar(X,Y);
plt.subplot(122); plt.plot(Y);

plt.subplot(211); plt.bar(X,Y);
plt.subplot(212); plt.plot(Y);

fig,ax = plt.subplots(1,2,figsize=(15,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)

fig,ax=plt.subplots(2,2,figsize=(15,12))
ax[0,0].bar(X,Y); ax[0,1].pie(Y,labels=X)
ax[1,0].plot(Y); ax[1,1].plot(Y,'.-',linewidth=3);

####2.1.2.2 pandas绘图函数
BSdata['体重'].plot(kind='line');
BSdata['体重'].plot(kind='hist');
BSdata['体重'].plot(kind='box');
BSdata['体重'].plot(kind='density');

BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')


T1=BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})
T1.plot(kind='bar'); #T1.sort_values().plot(kind='bar');
T1.plot(kind='pie');

##2.2 数据的分组分析
###2.2.1 频数分析
####2.2.1.1 计数数据的频数分析
#（1）pivot_table
BSdata['开设'].value_counts()
BSdata['开设'].value_counts().plot(kind='bar')

BSdata.pivot_table(values='学号',index='开设',aggfunc=len)
T1=BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})
T1.plot(kind='bar');
T1.plot(kind='pie');
pd.pivot_table(BSdata,values='学号',index='开设',aggfunc=len)
#BSdata.pivot_table(values='学号',index='开设',aggfunc=len)

####2.2.1.2 计量数据的频数分析
#（1）身高频数表
pd.cut(BSdata.身高,bins=10).value_counts()
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar');

#（2）支出频数表
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar');

###2.2.2 列联表分析
####2.2.2.1 计数数据的列联表
#（1）二维列联表
pd.crosstab(BSdata.开设,BSdata.课程)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)

pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all')

BSdata.pivot_table('学号','开设','课程',aggfunc=len)
BSdata.pivot_table('学号',index='开设',columns='课程',aggfunc=len)
pd.pivot_table(BSdata,values='学号',index='开设',columns='课程',aggfunc=len)

#（2）复式条图
T2=pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar');
T2.plot(kind='barh');
T2.plot(kind='bar',stacked=True);

####2.2.2.2 计量数据的列联表
#（1）groupby函数
BSdata1=BSdata.iloc[:,2:5];BSdata1.head()
BSdata1.mean()             #对计量数据求均值
BSdata.groupby(['性别'])
type(BSdata.groupby(['性别']))
BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean()

#（2）agg函数
BSdata.groupby(['性别'])['身高'].agg([np.mean, np.std])

#（3）应用apply()
BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)


###2.2.3 透视表分析
####2.2.3.1 计数数据的透视分析
#（1）pivot_table
BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['性别','开设'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['开设'],columns=['性别'],aggfunc=len)

####2.2.3.2 计量数据的透视分析
#pd.pivot_table(BSdata,index=["性别"],aggfunc=len)
BSdata.pivot_table(index=['性别'],values=["身高"],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=["身高"],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=["性别"],values=["身高","体重"])

#2.2.3.3 复合数据的透视分析
#pd.pivot_table(BSdata,index=["性别","开设"],aggfunc=len,margins=True)
#pd.pivot_table(BSdata,index=["性别"],aggfunc=np.mean)
BSdata.pivot_table('学号',['性别','开设'],'课程',aggfunc=len,margins=True,margins_name='合计')
BSdata.pivot_table(['身高','体重'],['性别',"开设"],aggfunc=[len,np.mean,np.std] )


#第3章 简单数据的统计分析
##3.1 随机变量及其分布
###3.1.1 随机变量及其分布
####3.1.1.1 均匀分布
a=0;b=1;y=1/(b-a)
plt.plot(a,y); #plt.axhlines(y=0,a,b);
plt.vlines(0,0,1);plt.vlines(1,0,1);

#####（1）整数随机数
import random
random.randint(10,20)  #[10,20]上的随机整数

#####（2）实数随机数
random.uniform(0,1)    #[0,1]上的随机实数

#####（3）整数随机数列
import numpy as np
np.random.randint(10,21,9)  #[10,20]上的随机整数

#####（4）实数随机数列
np.random.uniform(0,1,10)   #[0,1]上的10个随机实数=np.random.rand(10)

####3.1.1.2 正态分布
#####（2）标准正态分布
from math import sqrt,pi   #调用数学函数，import math as *
x=np.linspace(-4,4,50);
y=1/sqrt(2*pi)*np.exp(-x**2/2);
plt.plot(x,y);

import scipy.stats as st  #加载统计方法包
P=st.norm.cdf(2);P

'''加载自定义库,在当前目录下建立PyDm1func.py函数库即可'''
import PyDm_fun as da
'''标准正态曲线面积（概率） '''
da.norm_p(-1,1)         #68.27%
da.norm_p(-2,2)         #94.45%
da.norm_p(-1.96,1.96)   #95%
da.norm_p(-3,3)         #99.73%
da.norm_p(-2.58,2.58)   #99%

za=st.norm.ppf(0.95);za   #单侧
[st.norm.ppf(0.025),st.norm.ppf(0.975)]  #双侧


#####（3）正态随机数
np.random.normal(10,4,5)  #产生5个均值为10标准差为4的正态随机数
np.random.normal(0,1,5)   #生成5个标准正态分布随机数

'''一页绘制四个正态随机图 '''
fig,ax = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        ax[i,j].hist(np.random.normal(0,1,500),bins = 50)
plt.subplots_adjust(wspace = 0,hspace=0)

z=np.random.normal(0,1,100)
plt.hist(z)

st.probplot(BSdata.身高, dist="norm", plot=plt); #正态概率图
st.probplot(BSdata['支出'], dist="norm", plot=plt);

###3.1.2 正态分布
####3.1.2.1 基本概念
#####（1）简单随机抽样
np.random.randint(0,2,10)  #[0,2)上的10个随机整数
i=np.random.randint(1,53,6);i #抽取10个学生，[1,52]上的6个整数
BSdata.iloc[i]       #随机抽取的6个学生信息
BSdata.sample(6)    #直接抽取6个学生的信息

####3.1.2.2 统计量及其分布
#####（1）正态分布模拟
def norm_sim1(N=1000,n=10):    # n样本个数, N模拟次数（即抽样次数）
    xbar=np.zeros(N)            #模拟样本均值
    for i in range(N):          #[0,1]上的标准正态随机数及均值
       xbar[i]=np.random.normal(0,1,n).mean()
    sns.distplot(xbar,bins=50)  #plt.hist(xbar,bins=50)
    print(pd.DataFrame(xbar).describe().T)
norm_sim1()
norm_sim1(10000,30)
#sns.distplot(norm_sim1())  #plt.hist(norm_sim1())
#sns.distplot(norm_sim1(n=30,N=10000)) #plt.hist(norm_sim1(n=30,N=10000))

def norm_sim2(N=1000,n=10):
    xbar=np.zeros(N)
    for i in range(N):
       xbar[i]=np.random.uniform(0,1,n).mean()  #[0,1]上的均匀随机数及均值
    sns.distplot(xbar,bins=50)
    print(pd.DataFrame(xbar).describe().T)
norm_sim2()
norm_sim2(10000,30)

#sns.distplot(norm_sim2())             #plt.hist(norm_sim2())
#sns.distplot(norm_sim1(n=30,N=10000)) #plt.hist(norm_sim2(n=30,N=10000))

#####（3）t分布曲线
x=np.linspace(-4,4,50);x
yn=st.norm.pdf(x,0,1)
yt2=st.t.pdf(x,2)
yt10=st.t.pdf(x,10)
plt.plot(x,yn,'r-',x,yt2,'b.',x,yt10,'g-.');
plt.legend(["N(0,1)","t(2)","t(10)"]);

###3.1.3 随机模拟及其应用

#3.1.3.1 模拟大数定律
def Bernoulli(N=100):
    p=np.zeros(N)
    for n in range(1,N):
       f=np.random.randint(0,2,n)  #[0,1]
       m=sum(f)
       p[n]=m/n
    plt.plot(p);plt.hlines(0.5,1,N)
Bernoulli()
Bernoulli(1000)

#3.1.3.2 模拟方法求积分
from math import sqrt,pi,exp
def g(x):
    return (1/sqrt(2*pi))*exp(-x**2/2)
def I(n,a,b,g):
    x=np.random.uniform(0,1,n)
    return sum([(b-a)*g(a+(b-a)*y) for y in x])/n
I(10000,-1,1,g)

from scipy.integrate import quad
quad(g,-1,1)

##3.2 随机模拟及其应用
###3.2.1 随机模拟方法

###3.2.2 模拟大数定律
def Bernoulli(N=100):
    p=np.zeros(N)
    for n in range(1,N):
        f=np.random.randint(0,2,n) #[0,1]
        m=sum(f)
        p[n]=m/n
    plt.plot(p);plt.hlines(0.5,1,N)
Bernoulli()
Bernoulli(1000)

###3.2.3 模拟方法求积分
from math import sqrt,pi,exp
def g(x):
    return (1/sqrt(2*pi))*exp(-x**2/2)
def I(n,a,b,g):
    x=np.random.uniform(0,1,n)
    return sum([(b-a)*g(a+(b-a)*y) for y in x])/n
I(10000,-1,1,g)

from scipy.integrate import quad
quad(g,-1,1)

##3.3 单变量统计分析模型
###3.3.1 简单线性相关分析
####3.3.1.1 线性相关的概念
x=np.linspace(-4,4,20); e=np.random.randn(20) #随机误差
fig,ax=plt.subplots(2,2,figsize=(15,12))
ax[0,0].plot(x,x,'o')
ax[0,1].plot(x,-x,'o')
ax[1,0].plot(x,x+e,'o');
ax[1,1].plot(x,-x+e,'o');

####3.3.1.2 相关系数的计算
#####（1）散点图
x=BSdata.身高;y=BSdata.体重
plt.plot(x, y,'o'); #plt.scatter(x,y);

#####（2）相关系数
x.cov(y)

x.corr(y)
y.corr(x)

####3.3.1.3 相关系数的检验
#####(3) 计算值和值，作结论。
st.pearsonr(x,y)  #pearson相关及检验

###3.3.2 简单线性回归分析
####3.3.2.1一元线性回归模型的估计
#####（1）模拟直线回归模型
dm.reglinedemo()

import statsmodels.api as sm             #简单线性回归模型
fm1=sm.OLS(y,sm.add_constant(x)).fit()   #普通最小二乘，家常数项
fm1.params                               #系数估计
yfit=fm1.fittedvalues;
plt.plot(x, y,'.',x,yfit, 'r-');
####3.3.2.2 一元线性回归模型的检验
#####
fm1.tvalues                            #系数t检验值
fm1.pvalues                            #系数t检验概率
pd.DataFrame({'b估计值':fm1.params,'t值':fm1.tvalues,'概率p':fm1.pvalues})

import statsmodels.formula.api as smf  #根据公式建回归模型
fm2=smf.ols('体重~身高', BSdata).fit()
pd.DataFrame({'b估计值':fm2.params,'t值':fm2.tvalues,'概率p':fm2.pvalues})
fm2.summary2().tables[1]           #回归系数检验表
plt.plot(BSdata.身高,BSdata.体重,'.',BSdata.身高,fm2.fittedvalues,'r-');

####3.3.2.3 一元线性回归模型的预测
fm2.predict(pd.DataFrame({'身高': [178,188,190]}))   #预测

####3.3.2.4 分组拟合一元线性模型
smf.ols('体重~身高',BSdata[BSdata.性别=='男']).fit().summary2().tables[1]
smf.ols('体重~身高',BSdata[BSdata.性别=='女']).fit().summary2().tables[1]

#第4章 复杂数据的综合分析
##4.1 多变量线性相关与回归
###4.1.1 多变量间线性相关
####4.1.1.1 相关系数阵
#####（1）读取无标签数据
pd.read_excel('PyDm_data.xlsx','MVdata')[:5]
MVdata=pd.read_excel('PyDm_data.xlsx','MVdata',index_col=0);round(MVdata,3)
YXdata=pd.read_excel('PyDm_data.xlsx','MVdata',index_col=0);
YXdata.columns=['Y','X1','X2','X3','X4','X5','X6','X7'];round(YXdata,3)
round(YXdata.cov(),2)

round(YXdata.corr(),4)
#4.1.1.2 矩阵散点图
pd.plotting.scatter_matrix(YXdata);
#4.1.1.3 相关检验矩阵
da.mcor_test(YXdata)

#4.1.2 多变量线性回归模型
#4.1.2.2 多元线性回归参数估计
import statsmodels.formula.api as smf  #根据公式建回归模型
M1=smf.ols('Y~X1',YXdata).fit(); M1.params
M2=smf.ols('Y~X1+X2',YXdata).fit(); M2.params
M3=smf.ols('Y~X1+X2+X3',YXdata).fit(); M3.params
Ms=smf.ols('Y~X1+X2+X3+X4+X5+X6+X7',YXdata).fit(); Ms.params
#3.1.2.3 多元线性回归模型检验
M1.summary()
M2.summary()
M3.summary()
Ms.summary()

import matplotlib.pyplot as plt          	#加载基本绘图包
plt.rcParams['font.sans-serif']=['SimHei'];   	#KaiTi SimHei黑体
plt.rcParams['axes.unicode_minus']=False;	#正常显示图中负号

#4.1.2.4 多元线性回归模型评判
et=Ms.resid                    #模型Ms的残差et
plt.plot(et.values,'.');              #残差图
ro=et.corr(et.shift(1));ro #et.shift(1) =et-1

DW=2*(1-ro);DW

Ms.summary()                        #回归模型简表
Ms.summary2().tables[0]             #回归模型统计量
Ms.summary2().tables[1]             #回归系数检验表
Ms.summary2().tables[2]             #模型残差分析表
Ms.summary2().tables[2][3][0]       #DW值
Ms.summary2().tables[2][3][1]       #JB值
Ms.summary2().tables[2][3][2]       #JB概率
R2=Ms.summary2().tables[0][1][6];R2 #模型的决定系数R2
Ms.summary2().tables[0][3][0]       #adj.R2
from math import sqrt
R=sqrt(float(R2));R
from statsmodels.iolib.summary2 import summary_col
summary_col([M1,M2,M3,Ms])        #模型结果比较

#4.2 综合评价方法
#4.2.1 综合评价指标体系
#4.2.1.1 评价指标体系的构建
#4.2.1.2 评价指标的基本分析
#### 单变量排名
MVdata
GDP=pd.DataFrame(MVdata.生产总值);GDP
GDP['排序']=(-GDP).rank(); GDP #GDP['排序']=GDP.rank(ascending=False);
#### 单多量排名
(-MVdata).rank()  #MVdata.rank(ascending=False)
#4.2.2 综合评价分析方法
#4.2.2.1 指标的无量纲化
#标准化法
def bz(x): return (x-x.mean())/x.std() #bz=lambda x: (x-x.mean())/x.std()
BZ=MVdata.apply(bz,0);  #BZ=(MVdata-MVdata.mean())/MVdata.std()
round(BZ,3)
#规范化法
#gf=lambda x: (x-x.min())/(x.max()-x.min())
def gf(x): return (x-x.min())/(x.max()-x.min())
GF=MVdata.apply(gf,0); #GF=(MVdata-MVdata.min())/(MVdata.max()-MVdata.min())
round(GF,3)
#4.2.2.2 简单平均评价法
#建立得分与排名数据框
SR=pd.DataFrame();SR
SR['BZscore']=BZ.mean(axis=1);
SR['BZrank']=(-SR.BZscore).rank(); SR
SR['GFscore']=GF.mean(1); #SR['GFscore']=GF.apply(np.mean,1)
SR['GFrank']=(-SR.GFscore).rank(); SR
#4.2.2.3 加权综合分析法
#变异系数法
CV=MVdata.std()/MVdata.mean();CV.T  #变异系数
W=CV/sum(CV);W                   #权重
SR['CVscore']=np.dot(BZ,W)
SR['CVrank']=SR.CVscore.rank(ascending=False); SR

#4.3 数据压缩方法
#4.3.1 主成分的基本思想
#4.3.2主成分的基本分析
#4.3.2.1 主成分分析步骤
Z=(MVdata-MVdata.mean())/MVdata.std()
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(Z)
Vi=pca.explained_variance_;Vi        #方差
Wi=pca.explained_variance_ratio_;Wi  #贡献率
Wi.sum()                             #累计贡献率
pd.DataFrame(pca.components_.T)      #主成分负荷
#4.3.2.2 主成分综合评价
import PyDm_fun as da
Si=pca.fit_transform(Z);Si           #主成分得分
Si=pd.DataFrame(Si,columns=['Comp1','Comp2'],index=MVdata.index);Si
plt.plot(Si.Comp1,Si.Comp2,'.')
da.hvline(Si.Comp1,Si.Comp2,Si.index);

Si['Comp']=Si.dot(Wi);Si          #综合得分
Si['Rank']=(-Si.Comp).rank();Si   #综合排名

#%run PyDm_fun.py
da.PCrank(MVdata,m=2)   #自定义主成分综合评价函数

#4.4 聚类分析方法
#4.4.1 聚类分析概念
#4.4.1.1 聚类分析法的思想
#4.4.1.2 聚类分析法的类型
#4.4.1.3 聚类分析的统计量
X12=YXdata[['X1','X2']][:11];X12  #取变量X1和X2前11个数据
plt.plot(X12.X1,X12.X2,'.')
for i in range(11):
    plt.text(X12.X1[i],X12.X2[i],X12.index[i])

Z12=(X12-X12.mean())/X12.std()         #数据标准化
import scipy.cluster.hierarchy as sch  #加载系统聚类包
D12=sch.distance.pdist(Z12);np.round(D12,3)        #样品间距离
Y12=pd.DataFrame(sch.distance.squareform(D12))  #距离矩阵
Y12.index=X12.index; Y12.columns=X12.index
round(Y12,3)  #输出距离阵

#4.4.2 系统聚类方法
#4.4.2.1 系统聚类的基本思想
H1=sch.linkage(D12);H1   #系统聚类过程，默认方法='complete'
sch.dendrogram(H1,labels=X12.index);  #系统聚类图
pd.DataFrame(sch.cut_tree(H1),index=X12.index)  #聚类划分

H2=sch.linkage(D12,method='ward');H2 #系统聚类过程，方法='ward'
sch.dendrogram(H2,labels=X12.index);

pd.DataFrame(sch.cut_tree(H2),index=X12.index)  #聚类划分

#4.4.2.2 系统聚类的基本步骤
Z=(MVdata-MVdata.mean())/MVdata.std()
D=sch.distance.pdist(Z);
H=sch.linkage(D,method='ward');
plt.figure(figsize=(10,6));                   #图形大小
sch.dendrogram(H,labels=MVdata.index);
pd.DataFrame(sch.cut_tree(H),index=MVdata.index).iloc[:,-5:]+1  #分5到1类
pd.DataFrame(sch.cut_tree(H),index=MVdata.index).iloc[:,27]+1   #分三类

#第5章 时序数据的模型分析
#5.1 时序数据的动态分析
#5.1.1 时间序列的介绍
#5.1.1.1 时间序列的概念
#####(1) 平稳序列模拟---随机游走
n=1000
rd=np.random.randn(n)
plt.plot(rd);
#####(2) 非平稳序列模拟---布朗运动
plt.plot(rd.cumsum())
#####(3) 平稳时间序列
#rd_ts=pd.Series(rd,index=pd.period_range('2001-01-01','2003-12-30'))
rd_ts=pd.Series(rd,index=pd.period_range('2001-01-01',periods=n));rd_ts
rd_ts.plot(grid=True);
#####(4) 非平稳时间序列
rd_ts.cumsum().plot(grid=True);

#5.1.3 时间序列的读取
TSdata=pd.read_excel('PyDm_data.xlsx','TSdata',index_col=0);TSdata.head()
TSdata.plot();
#Close['2016'].plot()

#5.3.1.3 股票收益率分析
def Return(Yt):   #计算收益率
    Rt=Yt/Yt.shift(1)-1  #Yt.diff()/Yt.shift(1)
    return(Rt)

Rt=Return(TSdata);Rt
Rt.plot().axhline(y=0);

#5.2 时间序列分析模型
#5.2.1 AR模型
np.random.seed(12)   #种子数，确保每次模拟结果一样
n=100
y1=np.zeros(n);y1
u=np.random.randn(n);u
for t in range(2,n):
    y1[t]=0.8*y1[t-1]+u[t]
plt.plot(y1,'o-')

#5.2.2 MA模型
np.random.seed(123)
y2=np.zeros(n);
u=np.random.randn(n);
for t in range(2,n):
    y2[t]=u[t]+0.6*u[t-1]
plt.plot(y2,'o-')

#5.2.3 ARMA模型
np.random.seed(123)
y3=np.zeros(n);
u=np.random.randn(n);
for t in range(2,n):
    y3[t]=0.8*y3[t-1]+u[t]+0.6*u[t-1]
plt.plot(y3,'o-');

#5.2.4 ARIMA模型
np.random.seed(12)
n=100
y4=np.random.randn(n).cumsum()
plt.plot(y4,'o-')
dy4=np.diff(y4)
plt.plot(dy4,'o-')
plt.plot(y4,'o-',dy4,'*-');plt.axhline(0);

#5.3 ARMA模型
##5.3.1 序列的相关性检验
from statsmodels.graphics.tsaplots import acf,plot_acf
np.round(acf(y2),3)

plot_acf(y1); # MR(1)模型的自相关系数

def ac_QP(Yt):
    import statsmodels.api as sm
    r,q,p = sm.tsa.acf(Yt, qstat=True)
    rqp=np.c_[r[1:], q, p]
    rqp=pd.DataFrame(rqp, columns=["AC", "Q", "Prob(>Q)"]);
    return(rqp)
ac_QP(y2)[:10]

from statsmodels.graphics.tsaplots import pacf,plot_pacf
np.round(pacf(y1),3)

plot_pacf(y2); # AR(1)模型的自相关系数

##5.3.2 ARMA 模型建立与检验
plot_acf(y3);
plot_pacf(y3);

import statsmodels.tsa.stattools as ts
ts.arma_order_select_ic(y1,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])
ts.arma_order_select_ic(y1,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])
ts.arma_order_select_ic(y3,max_ar=3,max_ma=3,ic=['aic', 'bic','hqic'])

from statsmodels.tsa.arima_model import ARMA
y1_arma=ARMA(y1,order=(1,0)).fit()
y1_arma.summary()

ARMA(y2,order=(0,1)).fit().summary()

ARMA(y3,order=(1,1)).fit().summary()

plt.plot(y3,'o-',ARMA(y3,order=(1,1)).fit().fittedvalues);

##5.4.3 序列的平稳性检验
from statsmodels.tsa.stattools import adfuller
def ADF(ts): #平稳性检验
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

round(ADF(y4),4)
round(ADF(dy4),4)

ADF(y1)
ADF(y2)
ADF(y3)

#5.4 股票指数预测模型的构建
Ct=TSdata['2015-04':'2018-04'].Close;
Ct.plot()

ADF(Ct)

plot_acf(Ct,lags=50);
plot_pacf(Ct,lags=50);

import statsmodels.tsa.stattools as ts
ts.arma_order_select_ic(Ct,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])

from statsmodels.tsa.arima_model import ARMA
Ct_ARMA=ARMA(Ct,order=(3,0)).fit()
Ct_ARMA.summary()

plt.plot(Ct,'o-',Ct_ARMA.fittedvalues);

Ct_05=pd.DataFrame({' 实际值':TSdata['2018-05'].Close}); #2018-05 收盘价数据
Ct_05[' 预测值']=Ct_ARMA.forecast(22)[0] # 模型预测数据
Ct_05[' 绝对误差']=Ct_05[' 实际值']-Ct_05[' 预测值'];
Ct_05[' 相对误差(%)']=Ct_05[' 绝对误差']/Ct_05[' 实际值']*100;
Ct_05

# 第6章 大数据分析简介 
## 6.1 大数据的概念

## 6.2 Python文本预处理
### 字符串的基本操作
#### 字符串的统计
len('abc')
S=["asfef", "qwerty", "yuiop", "b", "stuff.blah.yech"];
len(S)
[len(s) for s in S]

##### 字符串连接与拆分
'Python'+' '+'Data Analysis'
'暨南大学'+'管理学院'

website = '%s%s%s' % ('Python', 'tab', '.com');website

listStr = ['Python', 'tab', '.com']
website = ''.join(listStr);website

S1='历史阐释;;历史事实;;历史评价;;唯物史观'
S1.split(';;')

S2='查理曼;;钦差巡察;;加洛林帝国;;法兰克;;中世纪'
S3='南宋;;政治忌讳;;人物评价;;人际关系;;包容政治'
S4=[S1,S2,S3];S4

def list_split(content,sep):
    new_list=[]
    for i in range(len(content)):
        new_list.append(list(filter(None,content[i].split(sep))))
    return new_list

list_split(S4,';;')

####  字符串查询与替换
#####  字符串查询
S5=['广州大学广州发展研究院','暨南大学文学院历史系','暨南大学管理学院']
'暨南大学' in S5[1]

def find_words(content,pattern):
    return [content[i] for i in range(len(content)) if (pattern in content[i]) == True]

find_words(S5,'暨南大学')
len(find_words(S5,'暨南大学'))
len(find_words(S5,'a'))

# ##### 字符串替换
'apple,orange'.replace("apple","banana")

def list_replace(content,old,new):
    return [content[i].replace(old,new) for i in range(len(content))]
S5=['广州大学广州发展研究院','暨南大学文学院历史系','暨南大学管理学院']
list_replace(S5,'暨南大学','华南农业大学')

%run PyDm_fun.py

## 网络爬虫技术应用
#### 读取网页
import requests
from bs4 import BeautifulSoup
# 链家二手房数据
url='https://gz.lianjia.com/ershoufang/pg'  #广州
# url='https://fs.lianjia.com/ershoufang/pg'  #佛山
#url='https://zs.ke.com/ershoufang/'           #中山

page=read_html(url)
soup=BeautifulSoup(page,'lxml')
houseInfo=html_text(soup,'.clear .title a');houseInfo   #.houseInfo
Price=html_text(soup,'.totalPrice span');Price

#计算运行时间
%time
LJdata=lianjia_all(url,3)

LJdata.to_excel('LJdata.xlsx',sheet_name='gz',index=False)

# #### 爬虫数据的统计分析
LJdata=pd.read_excel('PyDm_data.xlsx','LJdata');LJdata[:6]
LJdata.info()
Price=LJdata['房屋价格'];Price
P=Price.astype(float);P
P.describe()
plt.hist(P,bins=np.arange(0,1000,100));
### 房屋信息信息提取
House=LJdata['房屋信息'];
House=[House[i].replace(' ','') for i in range(len(House))]
House1=list_split(House,'|');House1[:6]

### 数据清洗：去除第7位NONA
length=[len(House1[i]) for i in range(len(House1))];
length_result=[idx for idx, e in enumerate(length) if e==7]
###去除独栋别墅
for i in length_result: House1[i].remove('独栋别墅')
###去除在朝向中的不协调表述
error_check=[i for i in range(len(House1)) if House1[i][1]=='联排别墅' or House1[i][1]=='独栋别墅']
for i in error_check: del House1[i][1]
House1[-6:]

### 构建分析用数据框
House2=pd.DataFrame(House1)
House2.info()
House2.columns=['小区','格式','面积','朝向','装修','电梯']
House2.head()
House2['小区'].value_counts()
House2['格式'].value_counts().plot(kind='barh');
House2['面积'].value_counts()
House2['朝向'].value_counts()
House2['装修'].value_counts()
House2['电梯'].value_counts()
MJ=House2['面积'].str[:-3].astype(float)
import PyDm_fun as da
da.freq(MJ,bins=[0,50,80,100,150,200])

#'''去除None'''
#dianti=list(House2.电梯)
#none_solve=[i for i in range(len(dianti)) if dianti[i]==None]
#len(none_solve)
#House2.iloc[10].电梯== None
#for i in none_solve:
#    House2.iloc[i].电梯='暂无数据'
#da.tab(House2.电梯)

## 数据库技术及应用
###  Python中数据库使用
###  数据库的建立与分析
###  Sqlite数据框的建立
from sqlalchemy import create_engine
engine=create_engine('sqlite:///LJdata.db')
LJdata.to_sql('LJdata',engine,index=False)
# ####  Sqlite3数据的处理
from sqlalchemy import create_engine
engine=create_engine('sqlite:///LJdata.db')
LJ=pd.read_sql('LJdata',engine)

LJ.info
LJ.columns
LJ.sex.value_counts().plot(kind='bar')
tab(LJ.sex,plot=True)

# 第7章 文献计量与知识图谱
## 文献计量研究的框架

# ## 文献数据的收集与分析
# ### 文献数据的获取
# ### 文献数据的收集
# #### 文献数据的读取
WXdata=pd.read_excel('PyDm_data.xlsx','WXdata');
#WXdata.columns
WXdata.info()
WXdata.shape
WXdata.iloc[:,:4].head()
WXdata.tail()

# ### 文献数据的分析
# #### 科研单位与基金统计
university=pd.read_excel('PyDm_data.xlsx','university');
university.学校名称.head()
fund=pd.read_excel('PyDm_data.xlsx','fund');
fund.基金名称.head()

def find_words(content,pattern):  #寻找关键词
    return [content[i] for i in range(len(content)) if (pattern in content[i]) == True]

def search_university(content,pattern):
    return len([find_words(content[i],pattern) for i in range(len(content)) if find_words(content[i],pattern) != []])


def list_split(content,separator):  #分解信息
    new_list=[]
    for i in range(len(content)):
        new_list.append(list(filter(None,content[i].split(separator))))
    return new_list

organ=list_split(WXdata['Organ'],';')
len(organ)
organ[0:5]

data1=pd.DataFrame([[i,search_university(organ,i)] for i in university['学校名称']])
data1.rename(columns={0:'学校名称',1:'频数'},inplace=True)
data1.sort_values(by='频数',ascending = False)[:10]

jijin=list_split(WXdata['Fund'].dropna(axis=0,how='all').tolist(),';;')
data2=pd.DataFrame([[i,search_university(jijin,i)] for i in fund['基金名称']])
data2.rename(columns={0:'学校名称',1:'频数'},inplace=True)
data2.sort_values(by='频数',ascending = False)[:10]

# #### 作者和关键词统计
keyword=list_split(WXdata['Keyword'].dropna(axis=0,how='all').tolist(),';;')
keyword1=sum(keyword,[])
pd.DataFrame(keyword1)[0].value_counts()[:10]

def list_replace(content,old,new): #清楚信息中的空格
    return [content[i].replace(old,new) for i in range(len(content))]

author=list_replace(WXdata['Author'].dropna(axis=0,how='all').tolist(),',',';')
author1=list_split(author,';');author1
type(author1)
author2=sum(author1,[])
type(author2)
pd.DataFrame(author2)[0].value_counts()[:10]

# #### 年份和期刊统计
WXdata.Source.value_counts()[:10]
WXdata.Year.value_counts().plot(kind='barh')

# ## 知识图谱和科研管理
# ### 科研管理评价
NKYWX=pd.read_excel('PyDm_data.xlsx','NKYWX');
NKYWX.shape
NKYWX.columns
NKYWX.iloc[:,:2].head()

NKYDW=pd.read_excel('PyDm_data.xlsx','NKYDW');
NKYDW.head()
#fund=pd.read_excel('PyDm_data.xlsx','fund');
#fund.基金名称.head()

organ=list_split(NKYWX['Organ'],';')
data1=pd.DataFrame([[i,search_university(organ,i)] for i in NKYDW['单位']])
data1.rename(columns={0:'单位',1:'频数'},inplace=True)
data1.sort_values(by='频数',ascending = False)[:8]

jijin=list_split(NKYWX['Fund'].dropna(axis=0,how='all').tolist(),';;')
data2=pd.DataFrame([[i,search_university(jijin,i)] for i in fund['基金名称']])
data2.rename(columns={0:'学校名称',1:'频数'},inplace=True)
data2.sort_values(by='频数',ascending = False)[:12]

author=list_replace(NKYWX['Author'].dropna(axis=0,how='all').tolist(),',',';')
author1=list_split(author,';')
author2=sum(author1,[])
pd.DataFrame(author2)[0].value_counts()[:5]

keyword=list_split(NKYWX['Keyword'].dropna(axis=0,how='all').tolist(),';;')
keyword1=sum(keyword,[])
pd.DataFrame(keyword1)[0].value_counts()[:5]

NKYWX.Source.value_counts()[:5]

#第8章 社会网络分析方法
## 社会网络的初步印象
### 社会网络分析概念
### 社会网络分析库
## 社会网络图的构建
### 社会网络数据形式
#### 以连线的形式构建网络
import networkx as nx
nG=nx.Graph();nG
nG.add_node('JFK')
nG.add_nodes_from(['SFO','LAX','ATL','FLO','DFW','HNL'])
nG.number_of_nodes()

nG.add_edges_from([('JFK', 'SFO'), ('JFK', 'LAX'), ('LAX', 'ATL'),('FLO','ATL'),('ATL','JFK'),('FLO','JFK'),('DFW','HNL')])
nG.add_edges_from([('OKC','DFW'),('OGG','DFW'),('OGG','LAX')])
nG.number_of_edges()

nG.nodes()
nG.edges()

nx.draw(nG, with_labels=True)

# #### 以矩阵的形式构建网络
NXdata=pd.read_excel('PyDm_data.xlsx','NXdata',index_col=0)
NXdata

nf=nx.from_pandas_adjacency(NXdata)
nx.draw(nf,with_labels=True)

# #### 社会网络图的布局
nx.draw(nG,pos=nx.circular_layout(nG), with_labels=True)
nx.draw(nG,pos=nx.kamada_kawai_layout(nG), with_labels=True)
nx.draw(nG,pos=nx.random_layout(nG), with_labels=True)
nx.draw(nG,pos=nx.spectral_layout(nG), with_labels=True)

### 网络统计量
#### 网络汇总描述
nx.info(nG)

# #### 密度
nx.density(nG)

# #### 直径
nx.diameter(nG)

#### 聚类系数与相邻节点
nx.transitivity(nG)
nx.clustering(nG)
list(nG.neighbors('ATL'))

# #### 中心性
nx.degree_centrality(nG)
nx.betweenness_centrality(nG)
nx.closeness_centrality(nG)

# #### 最短路径
len(nx.shortest_path(nG,'ATL','SFO'))

# ### 知识图谱应用
# #### 图谱共现矩阵
# #### 共显矩阵网络图
def occurence(data,document):  #生成共现矩阵
    empty1=[];empty2=[];empty3=[]
    for a in data:
        for b in data:
            count = 0
            for x in document:
                if  [a in i for i in x].count(True) >0 and [b in i for i in x].count(True) >0:
                        count += 1
            empty1.append(a);empty2.append(b);empty3.append(count)
    df=pd.DataFrame({'from':empty1,'to':empty2,'weight':empty3})
    G=nx.from_pandas_edgelist(df, 'from', 'to', 'weight')
    return (nx.to_pandas_adjacency(G, dtype=int))

##提取上章文献数据的高频数据
organ=list_split(WXdata['Organ'],';')
data1=pd.DataFrame([[i,search_university(organ,i)] for i in university['学校名称']])
data1.rename(columns={0:'学校名称',1:'频数'},inplace=True)
keyword=list_split(WXdata['Keyword'].dropna(axis=0,how='all').tolist(),';;')
keyword1=sum(keyword,[])
author=list_replace(WXdata['Author'].dropna(axis=0,how='all').tolist(),',',';')
author1=list_split(author,';')
author2=sum(author1,[])

#获取前30名的高频数据
data_author=pd.DataFrame(author2)[0].value_counts()[:30].index.tolist();data_author
data_keyword=pd.DataFrame(keyword1)[0].value_counts()[0:30].index.tolist();data_keyword
data_university=data1.sort_values(by='频数',ascending = False)[0:30]['学校名称'].tolist()

Matrix1=occurence(data_author,author1);Matrix1
Matrix2=occurence(data_university,organ)
Matrix3=occurence(data_keyword,keyword)

import networkx as nx
graph1=nx.from_pandas_adjacency(Matrix1)
nx.draw(graph1,with_labels=True,node_color='yellow')

graph2=nx.from_pandas_adjacency(Matrix2)
nx.draw(graph2,with_labels=True,node_color='yellow')
graph3=nx.from_pandas_adjacency(Matrix3)
nx.draw(graph3,with_labels=True,node_color='yellow')

import scipy.cluster.hierarchy as sch
H1=sch.linkage(Matrix3,method='ward');
sch.dendrogram(H1,labels=Matrix3.index,orientation='right');

## Load R dataset
#import statsmodels.api as sm
#sm.datasets.get_rdataset("datasets", "USJudgeRatings").data
#sm.datasets.get_rdataset("cluster.datasets", "all.us.city.crime.1970").data
