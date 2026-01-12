对ELBO推倒的另一种方法，利用凸函数的相关特性
![[B-1.png]]
凸函数定义:对任意两点$x_1$和$x_2$，若$\phi$在0和1之间，且下面的条件成立，则该函数为“凸函数”:
$$
\phi f(x_1) + (1-\phi)f(x_2) \geq f(\phi x_1 + (1-\phi)x_2)
$$
![[B-3.png]]

> 是否为凸函数也可以由二阶导确认，二阶导应该都大于等于0

扩展到多个内分点，就是Jensen不等式。

假设有N个正实数{$\phi_1,\phi_2,\phi_3,...,\phi_N$}且满足$\sum_{n=1}^N \phi_n = 1$那么当$f(x)$时，满足以下Jesen不等式：
$$
\sum_{n=1}^N \phi_n f(x_n) \geq f(\sum_{n=1}^N \phi_n  x_n)
$$
- $\phi$是权重
- 左边：权重乘以函数值的求和
- 右边：xn乘以权重再放到函数里

## 凹函数和log函数
- 凹函数：与凸函数特性完全相反，所以凸函数带有负号就是凹函数。
- $-log x$是凸函数，所以

$$
\sum_{n=1}^N -\phi_n log(x_n) \geq -log\sum_{n=1}^N \phi_n  x_n
$$
两边乘以-1：
$$
\sum_{n=1}^N \phi_n log(x_n) \leq log\sum_{n=1}^N \phi_n  x_n
$$
## ELBO推导

假设q(z)是一个概率分布：
- $0\leq q(z)\leq 1$
- $\sum_z q(z) = 1$

SO:

$$
\sum_{x} q(z) logf(z) \leq log\sum_{z} q(z)  f(z)
$$
用$logp(x;\theta)$表示对数似然，然后用Jesen不等式将对数似然展开后：

$$
logp(x;\theta) = log\sum_z q(z)\frac{p(x,z;\theta)}{q(z)} \geq log\sum_{z} q(z) \frac{p(x,z;\theta)}{q(z)}
$$



88