# -*- coding: utf-8 -*-
"""
###   Python数据挖掘及应用
###   函数库 PyDm_fun.py
###   王斌会 王术 2019-1-6

#在函数末尾加上一个分号即可不输出结果
"""

##初始设置
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  #多项输出

import numpy as np                         #加载数组运算包
np.set_printoptions(precision=4)           #设置numpy输出精度
import pandas as pd                        #加载数据分析包
pd.set_option('display.width', 120)        #设置pandas输出宽度
pd.set_option('display.precision',4)       #设置pandas输出精度
import matplotlib.pyplot as plt            #加载基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi']; #SimHei黑体
plt.rcParams['axes.unicode_minus']=False;  #正常显示图中负号
#plt.figure(figsize=(5,4));                #图形大小
#plt.figure(figsize=(4,3), dpi=100, facecolor="white");
#plt.style.use(['bmh','wihte_background'])
#plt.style.use(['default'])
import scipy.stats as st

def tab(x):  #计数频数表
   f=x.value_counts();f
   s=sum(f);s
   p=round(f/s*100,3);p
   T1=pd.concat([f,p],axis=1);T1
   T1.columns=['例数','构成比'];T1
   T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
   Tab=T1.append(T2)
   return(round(Tab,3))

def freq(X,bins=10,density=False): #计量频数表与直方图
    if density:
        H=plt.hist(X,bins,density=density)
        plt.plot(H[1],st.norm.pdf(H[1]),color='r');
    else:
       H=plt.hist(X,bins);
    a=H[1][:-1];a
    b=H[1][1:];b
    f=H[0];f
    p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp],index=['[下限','上限)','频数','频率(%)','累计频数(%)'])
    return(round(Freq.T,2))

def stats(x): #基本统计量
    stat=[x.count(),x.min(),x.quantile(.25),x.mean(),x.median(),
           x.quantile(.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min', 'Q1(25%)','Mean','Median',
                               'Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    return(stat)
#x.plot(kind='kde')   #拟合的密度曲线，见下节
#plt.vlines(x.mean(),0,0.05,colors='b'); plt.vlines(x.median(),0,0.05,colors='r');
#import seaborn as sns
#sns.distplot(x, kde=True);

def norm_p(a=-2,b=2): ### 正态曲线面积图
    #a=-2;b=2;k=0.1
    x=np.arange(-4,4,0.1)
    y=st.norm.pdf(x)
    x1=x[(a<=x) & (x<=b)];x1
    y1=y[(a<=x) & (x<=b)];y1
    p=st.norm.cdf(b)-st.norm.cdf(a);p
    plt.title("N(0,1)分布: [%6.3f %6.3f] p=%6.4f"%(a,b,p))
    plt.plot(x,y);
    plt.hlines(0,-4,4);
    plt.vlines(x1,0,y1,colors='r');
    plt.text(-0.5,0.2,"p=%6.4f" % p,fontsize=15);

def t_p(a=-2,b=2,df=10,k=0.1):
    x=np.arange(-4,4,k)
    y=st.t.pdf(x,df)
    x1=x[(a<=x) & (x<=b)];x1
    y1=y[(a<=x) & (x<=b)];y1
    p=st.t.cdf(b,df)-st.t.cdf(a,df);p
    plt.plot(x,y);
    plt.title("t(%2d): [%6.3f %6.3f] p=%6.4f"%(df,a,b,p))
    plt.hlines(0,-4,4); plt.vlines(x1,0,y1,colors='r');
    plt.text(-0.5,0.2,"p=%6.4f" % p,fontsize=15);

from math import sqrt
def t_interval(b,x):
    a=1-b
    n = len(x)
    ta=st.t.ppf(1-a/2,n-1);ta
    se=x.std()/sqrt(n)
    return(x.mean()-ta*se, x.mean()+se*ta)
#t_interval(0.95,BSdata['身高'])

def ttest_1plot(X,mu=0): # 单样本均值t检验图
    k=0.1
    df=len(X)-1
    t1p=st.ttest_1samp(X, popmean = mu);t1p
    x=np.arange(-4,4,k)
    y=st.t.pdf(x,df)
    t=abs(t1p[0]);p=t1p[1]
    x1=x[x<=-t];x1
    y1=y[x<=-t];y1
    x2=x[x>=t];x2
    y2=y[x>=t];y2
    print("  单样本t检验\t t=%6.3f p=%6.4f"%(t,p))
    print("  t置信区间: ",st.t.interval(0.95,len(X)-1,X.mean(),X.std()))
    #plt.title("t检验图: t=%6.3f p=%6.4f"%(t,p),fontsize=15)
    plt.plot(x,y);
    plt.hlines(0,-4,4);
    plt.vlines(x1,0,y1,colors='r');
    plt.vlines(x2,0,y2,colors='r');
    #plt.text(-3.3,0.08,"p/2=%6.4f" % (t1p[1]/2),fontsize=15);
    #plt.text(2,0.08,"p/2=%6.4f" % (t1p[1]/2),fontsize=15);
    plt.text(-0.5,0.05,"p=%6.4f" % t1p[1],fontsize=15);
    plt.vlines(st.t.ppf(0.05/2,df),0,0.2,colors='b');
    plt.vlines(-st.t.ppf(0.05/2,df),0,0.2,colors='b');
    plt.text(-0.5,0.2,r"$\alpha$=%3.2f"%0.05,fontsize=15);

def reglinedemo(n=20):    #模拟直线回归
    x=np.arange(n)+1
    e=np.random.normal(0,1,n)
    y=2+0.5*x+e
    import statsmodels.api as sm
    x1=sm.add_constant(x);x1
    fm=sm.OLS(y,x1).fit();fm
    plt.plot(x,y,'.',x,fm.fittedvalues,'r-'); #添加回归线，红色
    for i in range(len(x)):
        plt.vlines(x,y,fm.fittedvalues,linestyles='dotted',colors='b');
#reglinedemo(30);   #最小二乘回归示意图---直线回归

def mcor_test(X):  #相关系数矩阵检验
    p=X.shape[1];p
    sp=np.ones([p, p]);sp
    for i in range(0,p):
        for j in range(i,p):
            sp[i,j]=st.pearsonr(X.iloc[:,i],X.iloc[:,j])[1]
            sp[j,i]=st.pearsonr(X.iloc[:,i],X.iloc[:,j])[0]
    R=pd.DataFrame(sp,index=X.columns,columns=X.columns)
    print(round(R,4))
    print("\n下三角为相关系数，上三角为概率")

def PCrank(X,m=2): #主成分评价函数
   from sklearn.decomposition import PCA
   Z=(X-X.mean())/X.std()
   p=Z.shape[1]
   pca = PCA(n_components=p).fit(Z)
   Vi=pca.explained_variance_;Vi
   Wi=pca.explained_variance_ratio_;Wi
   Vars=pd.DataFrame({'Variances':Vi});Vars  #,index=X.columns
   Vars.index=['Comp%d' %(i+1) for i in range(p)]
   Vars['Explained']=Wi*100;Vars
   Vars['Cumulative']=np.cumsum(Wi)*100;
   print("\n方差贡献:\n",round(Vars,4))
   Compi=['Comp%d' %(i+1) for i in range(m)]
   loadings=pd.DataFrame(pca.components_[:m].T,columns=Compi,index=X.columns);
   print("\n主成分负荷:\n",round(loadings,4))
   scores=pd.DataFrame(pca.fit_transform(Z)).iloc[:,:m];
   scores.index=X.index; scores.columns=Compi;scores
   scores['Comp']=scores.dot(Wi[:m]);scores
   scores['Rank']=scores.Comp.rank(ascending=False);scores
   print('\n综合得分与排名:\n',round(scores,4))
   plt.plot(scores.Comp1,scores.Comp2,'.');
   for i in range(Z.shape[0]):
       plt.text(scores.Comp1[i],scores.Comp2[i],X.index[i])
   plt.hlines(0,scores.Comp1.min(),scores.Comp1.max(),linestyles='dotted')
   plt.vlines(0,scores.Comp2.min(),scores.Comp2.max(),linestyles='dotted')
   #return(Vars,loadings)

def hvline(X,Y,Z,h=0,v=0,lty='dotted'):
    plt.hlines(h,X.min(),X.max(),linestyles=lty)
    plt.vlines(v,Y.min(),Y.max(),linestyles=lty)
    for i in range(Z.shape[0]):
        plt.text(X[i],Y[i],Z[i]);

import requests
def read_html(url,encoding='utf-8'):    #定义读取html网页函数
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36" }
    response =requests.get(url,headers=headers)
    response.encoding = 'utf-8'
    return(response.text)
    
def html_text(info,word): #按关键词解析文本
    return([w.get_text() for w in info.select(word)])

from bs4 import BeautifulSoup
def lianjia_page(Soup):  #单个网页信息
    lianjia=pd.DataFrame()
    lianjia['房屋信息']=html_text(Soup,'.clear .title a')  #.houseInfo    
    lianjia['房屋价格']=html_text(Soup,'.totalPrice span')
    lianjia['房屋位置']=html_text(Soup,'.positionInfo a')
    lianjia['房屋单价']=html_text(Soup,'.unitPrice span')
    return(lianjia)

def lianjia_all(url,long): #所有网页信息
    houseinfo=pd.DataFrame()
    for i in range(long):
        web=read_html(url+str(i))
        soup=BeautifulSoup(web,'lxml')
        pages=lianjia_page(soup)
        houseinfo=pd.concat([houseinfo,pages])
    return(houseinfo)

def find_words(content,pattern):  #寻找关键词
    return [content[i] for i in range(len(content)) if (pattern in content[i]) == True]

def search_university(content,pattern):
    return len([find_words(content[i],pattern) for i in range(len(content)) if find_words(content[i],pattern) != []])

def list_split(content,separator):  #分解信息
    new_list=[]
    for i in range(len(content)):
        new_list.append(list(filter(None,content[i].split(separator))))
    return new_list
    