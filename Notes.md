# Notes

# 1 Classification vs. Regression

## 1.1 Gradient Decent

找$J(\theta)$ 的最小值，$\theta\leftarrow\theta-\alpha\dfrac{\partial J}{\partial\theta},\;\alpha\stackrel{\triangle}{=}\text{learning rate}
$.

## 1.2 Regression (Linear Regression)

__– 预测连续值__

设$X=\left[\begin{matrix}x_{0}^{(1)},x_{1}^{(1)},\cdots x_{n}^{(1)}\\
\vdots\\
x_{0}^{(m)},x_{1}^{(m)},\cdots x_{n}^{(m)}
\end{matrix}\right](n\text{ features},m\text{ training datas})
, y\in\mathbb{R}^{m}=$ 训练数据答案

设 $\theta=\left[\begin{matrix}\theta_{0}\\
\vdots\\
\theta_{m}
\end{matrix}\right]$
（线性方程的参数）, $\theta$下的预测值为$p=X\cdot\theta$.

__cost function __

$ J(\theta)=\dfrac{1}{2m}\left\Vert y-p\right\Vert ^{2}$

$\dfrac{\partial J}{\partial\theta}=\dfrac{1}{m}X^{T}(p-y)$

__normal equation __

$\theta_{ans}=(XX^{T})^{-1}X^{T}y$，如果不可逆，说明有冗余features。

__feature scalling, __

$X^{(i)}\leftarrow X^{(i)}-\text{avg}; X^{(i)}\leftarrow\dfrac{X^{(i)}}{\text{std deviation or (max-min)}}$

## 1.3 Classification (Logistic Regression)

__— 预测离散值__

__预测0-1方法：__

Sigmoid Function $g:R\rightarrow(0,1),\;g(x)=\dfrac{1}{1+e^{-x}}=P(ans=1)=1-P(ans=0)$

let $h=g(X\theta)$

cost function $J(\theta)=-\dfrac{1}{m}\left(y^{T}\ln h+(1-y)^{T}\ln(1-h)\right)$

每个值$h^{(i)}$对cost function的贡献为$\begin{cases}
-\ln h^{(i)} & \text{if }y^{(i)}=1\\
-\ln(1-h^{(i)}) & \text{if }y^{(i)}=0
\end{cases}	$

$\dfrac{\partial J}{\partial\theta}=\dfrac{1}{m}X^{T}(h-y)$

normal equation $\theta_{ans}=????$

__预测n个值？__

__one vs. all__ 对于每个预测值$k$，在训练集训练$n$个classifier，训练第$k$个时将$ans=k
$当作1，其它当作0。作预测用$k$个classifier计算$P(ans=k)$，找出概率最大的$k$，即为答案。

## 1.4 Regularization

__– prevent over-fitting.__

cost function $J'(\theta)=J(\theta)+\dfrac{1}{2m}{\displaystyle \sum_{i=1}^{m}\theta_{i}^{2}}$

此时，linear regression normal equation $\theta=\left(X^{T}X+\lambda\cdot L\right)^{-1}X^{T}y
, \text{where}\; L=\begin{bmatrix}0\\
 & 1\\
 &  & \ddots\\
 &  &  & 1
\end{bmatrix}$

# 2 Neuron Network

– (k+1)-th layer $a^{(k+1)}=g(\Theta^{(k)}[1;\,a^{(k)}])$ (Forward 
propagation)

activation function: $g(z)=\dfrac{1}{1+e^{-z}}$

__– Back propagation__ (Calculate $\dfrac{\partial J(\Theta)}{\partial\Theta_{ij}^{(k)}}$
  )

$\delta_{i}^{(k)}\stackrel{def}{=}\dfrac{\partial J}{\partial z_{i}^{(k)}}
, (z_{i}^{(k)}=\text{第}k\text{层第}i\text{个节点经过激活函数之前的值})$

$\delta^{(k)}=\left(\Theta^{(k)}\right)^{T}\delta^{(k+1)}.*g'(a^{(k+1)}),\;\delta^{(L)}=\dfrac{\partial J}{\partial z^{(L)}}=\dfrac{\partial J}{\partial a^{(L)}}\cdot\dfrac{\partial a^{(L)}}{\partial z^{(L)}}=y-a^{(L)}$

## 2.1 Evaluating a learning algo.

为了选择最佳的多项式次数d，将训练集分成三个部分：60%Training set，20%Cross validation 
set，20%Test Set。

对于每个多项式系数d，我们先在Training set上学习出最佳的$\Theta^{(d)}。之后，选出J_{\text{cross}}(\Theta^{(d)})
最小的d。然后我们用J_{test}(\Theta^{(d)})$来测量这个模型的误差值。

为什么不直接用Test set选取最佳d？因为我们若选取使得$J_{\text{test}}(\Theta^{(d)})$最小的$d$
，可能使$J_{\text{test}}$过于小了（多fit了一个d参数）。

## 2.2 Bias vs. variance

Bias (underfit) problem – $J_{\text{train}}(\Theta)$ will be high, $J_{\text{CV}}(\Theta)\approx J_{\text{Train}}(\Theta)$

variance(overfit) problem – $J_{\text{Train}}(\Theta)$ will be low,$ 
J_{\text{CV}}\gg J_{\text{Train}}(\Theta)$

![grapi](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/r2LZ-dnoEeazeA79Dx1Wzg_7882490104e892f51a242615f3d71ba3_bias-variance.png?expiry=1490400000000&hmac=_hyIAYiUYnMAyXl3p8IckydYkkKJBKNk2-ajf15nkxk)

## 2.3 Regularization and Bias/Variance
与选择多项式次数$d$十分相似，但是cost function $J$ （对于train set， cv， test）
不计入$\lambda\sum_{}^{}{{\Theta^{(k)}_{ij}}^2}$
尝试的$\lambda$从0.01开始，每次乘2，找出最佳的$\lambda$。


1. Create a list of lambdas (i.e. λ∈{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
2. Create a set of models with different degrees or any other variants.
Iterate through the λs and for each λ go through all the models to learn some Θ.
3. Compute the cross validation error using the learned Θ (computed with λ) on the JCV(Θ) __without__ regularization or λ = 0.
4. Select the best combo that produces the lowest error on the cross validation set.
Using the best combo Θ and λ, apply it on Jtest(Θ) (__without regularization__) to see if it has a good generalization of the problem.

## 2.4 Learning Curves
If a learning algo is suffering from high bias, getting mre training data will not help much.

.................. is suffering from high variance, ......................is likely to help.

![Alt text](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bpAOvt9uEeaQlg5FcsXQDA_ecad653e01ee824b231ff8b5df7208d9_2-am.png?expiry=1490400000000&hmac=Ml6Sr85-algJ1oEy9Rg5eTphKQxQaKPLdWd_EVqTjKQ )
![Alt text](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/vqlG7t9uEeaizBK307J26A_3e3e9f42b5e3ce9e3466a0416c4368ee_ITu3antfEeam4BLcQYZr8Q_37fe6be97e7b0740d1871ba99d4c2ed9_300px-Learning1.png?expiry=1490400000000&hmac=UH-cHqWjUBBsI34NTRIEriaR9AA3INCXRvxMNMsWP7Y)

## Review
- Getting more training examples: Fixes high variance
- Trying smaller sets of features: Fixes high variance
- Adding features: Fixes high bias
- Adding polynomial features: Fixes high bias
- Decreasing λ: Fixes high bias
- Increasing λ: Fixes high variance.

__Diagnosing Neural Networks__

- A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
- A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase λ) to address the overfitting.
Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.
