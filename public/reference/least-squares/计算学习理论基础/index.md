# 计算学习理论基础

黄建国

(上海交通大学数学科学学院, 上海200240)

## 目 录

1 若干基本知识 2

2 Rademacher 复杂度 5

3 VC 维数 10

4 求解微分方程的二层神经网络算法误差分析 12

## 1 若干基本知识

引理 1.1 (Hoeffding不等式) 若 \( {X}_{1},{X}_{2},\cdots,{X}_{m} \) 为 \( m \) 个独立随机变量,且满足 \( 0 \leq  {X}_{i} \leq  1 \), 则对 \( \forall \varepsilon  > 0 \),成立Hoeffding 不等式

\[\mathbb{P}\left( {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} - \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {X}_{i}\right\rbrack   \geq  \varepsilon }\right)  \leq  {e}^{-{2m}{\varepsilon }^{2}}, \tag{1.1}\]

\[\mathbb{P}\left( {\left| {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} - \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {X}_{i}\right\rbrack  }\right|  \geq  \varepsilon }\right)  \leq  2{e}^{-{2m}{\varepsilon }^{2}}. \tag{1.2}\]

换言之,如果令 \( \delta  = \exp \left( {-{2m}{\varepsilon }^{2}}\right) \),则Hoeffding 不等式

\[\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} \leq  \mathbb{E}\left\lbrack  {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i}}\right\rbrack   + {\left( \frac{1}{2m}\ln \left( \frac{1}{\delta }\right) \right) }^{\frac{1}{2}} \tag{1.3}\]

至少以 \( 1 - \delta \) 的概率成立.

证明. 该结果证明取自文献 \( \left\lbrack  1\right\rbrack \),关键是证明(1.1). 事实上,若记 \( {X}_{i}^{\prime } = 1 - {X}_{i} \),则 \( {X}_{1}^{\prime },{X}_{2}^{\prime } \), \( \cdots,{X}_{m}^{\prime } \) 相互独立,且 \( 0 \leq  {X}_{i}^{\prime } \leq  1 \),故由(1.1)立知

\[\mathbb{P}\left( {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i}^{\prime } - \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {X}_{i}^{\prime }\right\rbrack   \geq  \varepsilon }\right)  \leq  {e}^{-{2m}{\varepsilon }^{2}},\]

此即

\[\mathbb{P}\left( {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} - \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {X}_{i}\right\rbrack   \leq   - \varepsilon }\right)  \leq  {e}^{-{2m}{\varepsilon }^{2}}.\]

所以，由(1.1)可知(1.2)也成立.

下面来证明(1.1). 记

\[{\mu }_{i} = \mathbb{E}\left\lbrack  {X}_{i}\right\rbrack ,\;{S}_{m} = \mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i},\;\mu  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{\mu }_{i}.\]

Step 1 获得Chernoff 不等式. 由Markov 不等式易知,对 \( \forall t > 0, s > 0 \),

\[\mathbb{P}\left( {z \geq  s}\right)  = \mathbb{P}\left( {{tz} \geq  {ts}}\right)  = \mathbb{P}\left( {{e}^{tz} \geq  {e}^{ts}}\right)  \leq  {e}^{-{ts}}\mathbb{E}\left\lbrack  {e}^{tz}\right\rbrack . \tag{1.4}\]

Step 2 获得以下辅助结果.

设随机变量 \( v \) 满足 \( \mathbb{E}\left\lbrack  v\right\rbrack   = 0 \) 且 \( a \leq  v \leq  b \) a.s.,则对 \( \forall t > 0 \),

\[\mathbb{E}\left\lbrack  {e}^{tv}\right\rbrack   \leq  {e}^{{t}^{2}{\left( b - a\right) }^{2}/8}. \tag{1.5}\]

事实上,因 \( f\left( x\right)  = \exp \left( {tx}\right) \) 为凸函数,所以

\[{e}^{tx} \leq  \frac{x - a}{b - a}{e}^{tb} + \frac{b - x}{b - a}{e}^{ta},\; a \leq  x \leq  b.\]

又 \( \mathbb{E}\left\lbrack  v\right\rbrack   = 0 \),由上有

\[\mathbb{E}\left\lbrack  {e}^{tv}\right\rbrack   \leq  \frac{b}{b - a}{e}^{ta} - \frac{a}{b - a}{e}^{tb}. \tag{1.6}\]

现设 \( p = b/\left( {b - a}\right), u = \left( {b - a}\right) t \). 考虑函数

\[\varphi \left( u\right)  = \ln \left( {p{e}^{ta} + \left( {1 - p}\right) {e}^{tb}}\right)\]

\[= {ta} + \ln \left( {p + \left( {1 - p}\right) {e}^{u}}\right)\]

\[= \left( {p - 1}\right) u + \ln \left( {p + \left( {1 - p}\right) {e}^{u}}\right).\]

由Taylor 展开知,存在 \( \xi  \in  \mathbb{R} \),使

\[\varphi \left( u\right)  = \varphi \left( 0\right)  + {\varphi }^{\prime }\left( 0\right) u + \frac{1}{2}{\varphi }^{\prime \prime }\left( \xi \right) {u}^{2}.\]

而由定义知

\[\varphi \left( 0\right)  = 0\]

\[{\varphi }^{\prime }\left( u\right)  = \left( {p - 1}\right)  + \frac{\left( {1 - p}\right) {e}^{u}}{p + \left( {1 - p}\right) {e}^{u}} = p - \frac{p}{p + \left( {1 - p}\right) {e}^{u}} \Rightarrow  {\varphi }^{\prime }\left( 0\right)  = 0,\]

\[{\varphi }^{\prime \prime }\left( u\right)  = \frac{p\left( {1 - p}\right) {e}^{u}}{{\left( p + \left( 1 - p\right) {e}^{u}\right) }^{2}} \Rightarrow  {\varphi }^{\prime \prime }\left( \xi \right)  \leq  \frac{1}{4}.\]

于是

\[\varphi \left( u\right)  \leq  \frac{{u}^{2}}{8} = \frac{1}{8}{\left( b - a\right) }^{2}{t}^{2}. \tag{1.7}\]

联立(1.6)-(1.7)立知(1.5).

Step 3 证明估计式(1.1). 由Chernoff不等式(1.4)知

\[P\left( {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} - \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {X}_{i}\right\rbrack   \geq  \varepsilon }\right)\]

\[= P\left( {{S}_{m} - {m\mu } \geq  {m\varepsilon }}\right)\]

\[\leq  {e}^{-{tm\varepsilon }}\mathbb{E}\left\lbrack  {e}^{t\left( {{S}_{m} - {m\mu }}\right) }\right\rbrack\]

\[= {e}^{-{tm\varepsilon }}\mathop{\prod }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {e}^{t{X}_{i} - t{\mu }_{i}}\right\rbrack .\]

可视 \( {\mu }_{i} \) 为固定数,于是 \( \mathbb{E}\left\lbrack  {{X}_{i} - {\mu }_{i}}\right\rbrack   = 0,{X}_{i} - {\mu }_{i} \in  \left\lbrack  {-{\mu }_{i},1 - {\mu }_{i}}\right\rbrack \),故由(1.5)知

\[\mathbb{E}\left\lbrack  {e}^{t{X}_{i} - t{\mu }_{i}}\right\rbrack   \leq  {e}^{{t}^{2}/8}.\]

从而易知

\[P\left( {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} - \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {X}_{i}\right\rbrack   \geq  \varepsilon }\right)  \leq  {e}^{-{tm\varepsilon } + m{t}^{2}/8}.\]

特取 \( t = {4m\varepsilon } \),上式右端取得最小值,即得 (1.1).

注 1.1 若 \( {\left\{  {X}_{i}\right\}  }_{i = 1}^{m} \) 彼此相互独立,且 \( {X}_{i} \in  \left\lbrack  {{a}_{i},{b}_{i}}\right\rbrack \), a.s.,则类似引理 1.1 的证明可知

\[\mathbb{P}\left( {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} - \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{E}\left\lbrack  {X}_{i}\right\rbrack   \geq  \varepsilon }\right)  \leq  {e}^{-2{m}^{2}{\varepsilon }^{2}/\mathop{\sum }\limits_{{i = 1}}^{m}{\left( {b}_{i} - {a}_{i}\right) }^{2}}.\]

注 1.2 若 \( {\left\{  {X}_{i}\right\}  }_{i = 1}^{m} \) 独立同分布,但没有有界性性质,则由 Chebyshev不等式可知

\[\mathbb{P}\left( {\left| {\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{X}_{i} - \mu }\right|  \geq  \varepsilon }\right)  \leq  \frac{{\sigma }^{2}}{m{\varepsilon }^{2}},\]

式中, \( \mu  = \mathbb{E}\left\lbrack  {X}_{1}\right\rbrack \) 而 \( {\sigma }^{2} = \operatorname{Var}\left\lbrack  {X}_{1}\right\rbrack \). 此时的结果明显不如前面的以指数量级衰减的结果.

Hoeffding 不等式是如下McDiarmid不等式的特例.

引理 1.2 (McDiarmid不等式) 若 \( {X}_{1},{X}_{2},\cdots,{X}_{m} \) 是 \( m \) 个相互独立的随机变量,且对任意 \( 1 \leq \; i \leq  m \),函数 \( f \) 满足

\[\mathop{\sup }\limits_{{{x}_{1},\cdots,{x}_{m},{\mathbf{x}}_{i}^{\prime }}}\left| {f\left( {{x}_{1},\cdots,{\mathbf{x}}_{i},\cdots,{x}_{m}}\right)  - f\left( {{x}_{1},\cdots,{\mathbf{x}}_{i}^{\prime },\cdots,{x}_{m}}\right) }\right|  \leq  {c}_{i},\]

则对任意 \( \varepsilon  > 0 \) ，有

\[\mathbb{P}\left( {f\left( {{X}_{1},\cdots,{X}_{m}}\right)  - \mathbb{E}\left\lbrack  {f\left( {{X}_{1},\cdots,{X}_{m}}\right) }\right\rbrack   \geq  \varepsilon }\right)  \leq  {e}^{-2{\varepsilon }^{2}/\mathop{\sum }\limits_{i}{c}_{i}^{2}}, \tag{1.8}\]

\[\mathbb{P}\left( {\left| {f\left( {{X}_{1},\cdots,{X}_{m}}\right)  - \mathbb{E}\left\lbrack  {f\left( {{X}_{1},\cdots,{X}_{m}}\right) }\right\rbrack  }\right|  \geq  \varepsilon }\right)  \leq  2{e}^{-2{\varepsilon }^{2}/\mathop{\sum }\limits_{i}{c}_{i}^{2}}. \tag{1.9}\]

证明. 同前一引理, 只需证明(1.8)即可. 记

\[{V}_{i} = \mathbb{E}\left\lbrack  {f \mid  {X}_{1},\cdots,{X}_{i}}\right\rbrack   - \mathbb{E}\left\lbrack  {f \mid  {X}_{1},\cdots,{X}_{i - 1}}\right\rbrack .\]

根据条件期望的性质易知, \( \mathbb{E}\left\lbrack  {{V}_{i} \mid  {X}_{1},\cdots,{X}_{i - 1}}\right\rbrack   = 0 \),且

\[f\left( {{X}_{1},\cdots,{X}_{m}}\right)  - \mathbb{E}\left\lbrack  {f\left( {{X}_{1},\cdots,{X}_{m}}\right) }\right\rbrack   = \mathop{\sum }\limits_{{i = 1}}^{m}{V}_{i}.\]

使用证明(1.5) 的方法,可知对 \( t > 0 \),

\[\mathbb{E}\left\lbrack  {{e}^{t{V}_{i}} \mid  {X}_{1},\cdots,{X}_{i - 1}}\right\rbrack   \leq  {e}^{{t}^{2}{c}_{i}^{2}/8}. \tag{1.10}\]

我们把这个结果的具体证明放在后面.

现在对任一 \( t > 0 \),由Markov 不等式有

\[\mathbb{P}\left( {f\left( {{X}_{1},\cdots,{X}_{m}}\right)  - \mathbb{E}\left\lbrack  {f\left( {{X}_{1},\cdots,{X}_{m}}\right) }\right\rbrack   \geq  \varepsilon }\right)\]

\[= \mathbb{P}\left( {\mathop{\sum }\limits_{{i = 1}}^{m}{V}_{i} \geq  \varepsilon }\right)  = \mathbb{P}\left( {{e}^{t\mathop{\sum }\limits_{{i = 1}}^{m}{V}_{i}} \geq  {e}^{t\varepsilon }}\right)\]

\[\leq  {e}^{-{t\varepsilon }}\mathbb{E}\left\lbrack  {e}^{t\mathop{\sum }\limits_{{i = 1}}^{m}{V}_{i}}\right\rbrack . \tag{1.11}\]

而由(1.10) 知

\[\mathbb{E}\left\lbrack  {e}^{t\mathop{\sum }\limits_{{i = 1}}^{m}{V}_{i}}\right\rbrack   = \mathbb{E}\left\lbrack  {{e}^{t\mathop{\sum }\limits_{{i = 1}}^{m}{V}_{i}} \mid  {X}_{1},\cdots,{X}_{m}}\right\rbrack\]

\[= \mathbb{E}\left\lbrack  {{e}^{t\mathop{\sum }\limits_{{i = 1}}^{{m - 1}}{V}_{i}} \cdot  \mathbb{E}\left\lbrack  {{e}^{t{V}_{m}} \mid  {X}_{1},\cdots,{X}_{m - 1}}\right\rbrack  }\right\rbrack\]

\[\leq  {e}^{{t}^{2}{c}_{m}^{2}/8} \cdot  \mathbb{E}\left\lbrack  {e}^{t\mathop{\sum }\limits_{{i = 1}}^{{m - 1}}{V}_{i}}\right\rbrack\]

......

\[\leq  {e}^{{t}^{2}\mathop{\sum }\limits_{{i = 1}}^{m}{c}_{i}^{2}/8}. \tag{1.12}\]

联立(1.11)-(1.12), 有

\[\mathbb{P}\left( {f\left( {{X}_{1},\cdots,{X}_{m}}\right)  - \mathbb{E}\left\lbrack  {f\left( {{X}_{1},\cdots,{X}_{m}}\right) }\right\rbrack   \geq  \varepsilon }\right)  \leq  {e}^{-{t\varepsilon } + {t}^{2}\mathop{\sum }\limits_{{i = 1}}^{m}{c}_{i}^{2}/8}.\]

特取 \( t = {4\varepsilon }/\mathop{\sum }\limits_{{i = 1}}^{m}{c}_{i}^{2} \) 即知结果.

下面简证(1.10). 记

\[{U}_{i} = \mathop{\sup }\limits_{u}\mathbb{E}\left\lbrack  {f \mid  {X}_{1: i - 1, u}}\right\rbrack   - \mathbb{E}\left\lbrack  {f \mid  {X}_{1: i - 1}}\right\rbrack\]

\[{L}_{i} = \mathop{\inf }\limits_{l}\mathbb{E}\left\lbrack  {f \mid  {X}_{1: i - 1, l}}\right\rbrack   - \mathbb{E}\left\lbrack  {f \mid  {X}_{1: i - 1}}\right\rbrack\]

式中 \( {X}_{1: i} = \left\lbrack  {{X}_{1},\cdots,{X}_{i}}\right\rbrack \). 则

\[{U}_{i} - {L}_{i} \leq  \mathop{\sup }\limits_{{l, u}}\mathbb{E}\left\lbrack  {f \mid  {X}_{1: i - 1, u}}\right\rbrack   - \mathbb{E}\left\lbrack  {f \mid  {X}_{1: i - 1, l}}\right\rbrack\]

\[\leq  \mathop{\sup }\limits_{{l, u}}{\int }_{{X}_{i + 1: m}}\left\lbrack  {f\left( {X}_{1: i - 1, u, i + 1: m}\right)  - f\left( {X}_{1: i - 1, l, i + 1: m}\right) }\right\rbrack  \mathop{\prod }\limits_{{j = i + 1}}^{m}{f}_{{X}_{j}}\left( {x}_{j}\right) \mathrm{d}{x}_{i + 1: m}\]

\[\leq  {c}_{i}{\int }_{{X}_{i + 1: m}}\mathop{\prod }\limits_{{j = i + 1}}^{m}{f}_{{X}_{j}}\left( {x}_{j}\right) \mathrm{d}{x}_{i + 1: m} = {c}_{i}\]

于是对 \( {V}_{i} \) 使用Hoeffding不等式即得(1.10).

## 2 Rademacher 复杂度

Rademacher 复杂度(Rademacher complexity) 是一种刻画空间复杂性的途径. 令 \( \mathcal{H} \) 表示假设空间,其中的假设是 \( \mathcal{X} \) 到 \( \mathcal{Y} = \{ 1, - 1\} \) 的映射,给定训练集 \( D = \left\{  {\left( {{x}_{1},{y}_{1}}\right),\cdots,\left( {{x}_{m},{y}_{m}}\right) }\right\}   \subset \; \mathcal{X} \),假设 \( h \) 的经验误差为

\[\widehat{E}\left( h\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\mathbb{I}\left( {h\left( {\mathbf{x}}_{i}\right)  \neq  {y}_{i}}\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\frac{1 - {y}_{i}h\left( {\mathbf{x}}_{i}\right) }{2} = \frac{1}{2} - \frac{1}{2m}\mathop{\sum }\limits_{{i = 1}}^{m}{y}_{i}h\left( {\mathbf{x}}_{i}\right),\]

假设 \( h \) 的泛化误差为

\[E\left( {h,\mathcal{D}}\right)  = {P}_{\left( {x, y}\right)  \sim  \mathcal{D}}\left( {h\left( x\right)  \neq  y}\right)  = {\mathbb{E}}_{\left( {x, y}\right)  \sim  \mathcal{D}}\left\lbrack  {\mathbb{I}\left( {h\left( x\right)  \neq  y}\right) }\right\rbrack ,\]

一般而言, \( \mathcal{D} \) 为一未知分布,研究有一定难度. 本节仅讨论经验误差,其中 \( \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{y}_{i}h\left( {\mathbf{x}}_{i}\right) \) 体现了预测值 \( h\left( {\mathbf{x}}_{i}\right) \) 和真实值 \( {y}_{i} \) 之间的一致性. 换言之,经验误差最小的假设满足

\[{\operatorname{argmax}}_{h \in  \mathcal{H}}\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{y}_{i}h\left( {\mathbf{x}}_{i}\right).\]

然而, \( {y}_{i} \) 受随机噪声影响,不再是 \( {\mathbf{x}}_{i} \) 的真实标签. 于是,考虑等概率取 \( \pm  1 \) 的随机变量 \( {\sigma }_{i} \),称其为Rademacher随机变量. 基于 \( {\sigma }_{i} \) 转而考虑 \( \mathop{\sup }\limits_{{h \in  \mathcal{H}}}\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}h\left( {\mathbf{x}}_{i}\right) \),再取期望而得

\[{\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{h \in  \mathcal{H}}}\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}h\left( {\mathbf{x}}_{i}\right) }\right\rbrack ,\]

其中 \( \mathbf{\sigma } = \left\lbrack  {{\sigma }_{1},\cdots,{\sigma }_{m}}\right\rbrack \). 有关更详细解释见参考文献 \( \left\lbrack  {2,3}\right\rbrack \).

定义 2.1 函数空间 \( \mathcal{F}: \mathcal{Z} \rightarrow  \mathbb{R} \) 关于给定集合 \( Z = \left\{  {{z}_{1},\cdots,{z}_{m}}\right\}   \subset  \mathcal{Z} \) 的经验 Rademacher 复杂度为

\[{\widehat{R}}_{Z}\left( \mathcal{F}\right)  = {\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}f\left( {z}_{i}\right) }\right\rbrack ,\]

而 \( \mathcal{F} \) 关于 \( \mathcal{Z} \) 在分布 \( \mathcal{D} \) 上的Rademacher复杂度为

\[{R}_{m}\left( \mathcal{F}\right)  = {\mathbb{E}}_{Z}\left\lbrack  {{\widehat{R}}_{Z}\left( \mathcal{F}\right) }\right\rbrack   \mathrel{\text{:= }} {\mathbb{E}}_{Z \subset  \mathcal{Z}: \left| Z\right|  = m}\left\lbrack  {{\widehat{R}}_{Z}\left( \mathcal{F}\right) }\right\rbrack .\]

在有些文献中，按以下方式引入 \( \mathcal{F} \) 关于 \( Z \) 的经验Rademacher复杂度.

定义 2.2 设 \( A \subset  {\mathbb{R}}^{m} \),定义

\[R\left( A\right)  = \frac{1}{m}{\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  A}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{a}_{i}}\right\rbrack ,\]

式中 \( \mathbf{a} = \left\lbrack  {{a}_{1},\cdots,{a}_{m}}\right\rbrack \). 则

\[R\left( {\mathcal{F} \circ  S}\right)  = {\widehat{R}}_{S}\left( \mathcal{F}\right),\]

式中 \( S = \left\{  {{z}_{1},\cdots,{z}_{m}}\right\} \),而 \( \mathcal{F} \circ  S = \left\{  {f\left( {z}_{1}\right),\cdots, f\left( {z}_{m}\right) : f \in  \mathcal{F}}\right\} \). 此处,将集合 \( S = \left\{  {{z}_{1},\cdots,{z}_{m}}\right\} \) 恒同于向量 \( \left\lbrack  {{z}_{1},\cdots,{z}_{m}}\right\rbrack \).

对于函数空间 \( \mathcal{F} = l \circ  \mathcal{H} = \{ z \rightarrow  l\left( {h, z}\right) : h \in  H\} \),给定 \( f \in  \mathcal{F}, S = \left\{  {{z}_{1},\cdots,{z}_{m}}\right\} \) 是一组关于分布 \( \mathcal{D} \) 的独立同分布抽样,定义

\[{L}_{\mathcal{D}}\left( f\right)  = {\mathbb{E}}_{\mathcal{D}}\left\lbrack  {f\left( z\right) }\right\rbrack ,\;{L}_{S}\left( f\right)  = \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}f\left( {z}_{i}\right),\]

\[{\operatorname{Rep}}_{\mathcal{D}}\left( {f, S}\right)  = \mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{\mathcal{D}}\left( f\right)  - {L}_{S}\left( f\right) }\right)\]

定理 2.1 成立

\[{\mathbb{E}}_{S}\left\lbrack  {{\operatorname{Rep}}_{\mathcal{D}}\left( {f, S}\right) }\right\rbrack   \leq  2{\mathbb{E}}_{S}\left\lbrack  {{\widehat{R}}_{S}\left( \mathcal{F}\right) }\right\rbrack . \tag{2.1}\]

证明.

Step 1 设 \( {S}^{\prime } = \left\lbrack  {{z}_{1}^{\prime },\cdots,{z}_{m}^{\prime }}\right\rbrack \) 为另一组关于分布 \( \mathcal{D} \) 的独立同分布抽样,易知 \( {L}_{\mathcal{D}}\left( f\right)  = \; {\mathbb{E}}_{{S}^{\prime }}\left\lbrack  {{L}_{{S}^{\prime }}\left( f\right) }\right\rbrack \). 又由 \( {S}^{\prime } \) 与 \( S \) 独立,有

\[{L}_{\mathcal{D}}\left( f\right)  - {L}_{S}\left( f\right)  = {\mathbb{E}}_{{S}^{\prime }}\left\lbrack  {{L}_{{S}^{\prime }}\left( f\right) }\right\rbrack   - {L}_{S}\left( f\right)  = {\mathbb{E}}_{{S}^{\prime }}\left\lbrack  {{L}_{{S}^{\prime }}\left( f\right)  - {L}_{S}\left( f\right) }\right\rbrack .\]

在上式两侧关于 \( f \in  \mathcal{F} \) 取上确界,并注意到随机变量期望的上确界小于等于随机变量上确界的期望, 即

\[\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathbb{E}\left\lbrack  f\right\rbrack   \leq  \mathbb{E}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}f}\right\rbrack\]

可知

\[\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{\mathcal{D}}\left( f\right)  - {L}_{S}\left( f\right) }\right)  = \mathop{\sup }\limits_{{f \in  \mathcal{F}}}{\mathbb{E}}_{{S}^{\prime }}\left\lbrack  {{L}_{{S}^{\prime }}\left( f\right)  - {L}_{S}\left( f\right) }\right\rbrack   \leq  {\mathbb{E}}_{{S}^{\prime }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{{S}^{\prime }}\left( f\right)  - {L}_{S}\left( f\right) }\right) }\right\rbrack .\]

再关于 \( S \) 取期望,则由上知

\[{\mathbb{E}}_{S}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{\mathcal{D}}\left( f\right)  - {L}_{S}\left( f\right) }\right) }\right\rbrack   \leq  \frac{1}{m}{\mathbb{E}}_{S,{S}^{\prime }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right\rbrack . \tag{2.2}\]

Step 2 考虑到对任一 \( j,{z}_{j}^{\prime } \) 和 \( {z}_{j} \) 是独立同分布的,故

\[{\mathbb{E}}_{S,{S}^{\prime }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {f\left( {z}_{j}^{\prime }\right)  - f\left( {z}_{j}\right)  + \mathop{\sum }\limits_{{i \neq  j}}^{m}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right) }\right\rbrack   = {\mathbb{E}}_{S,{S}^{\prime }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {f\left( {z}_{j}\right)  - f\left( {z}_{j}^{\prime }\right)  + \mathop{\sum }\limits_{{i \neq  j}}^{m}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right) }\right\rbrack .\]

而Rademacher随机变量 \( {\sigma }_{j} \) 满足 \( \mathbb{P}\left( {{\sigma }_{j} = 1}\right)  = \mathbb{P}\left( {{\sigma }_{j} =  - 1}\right)  = \frac{1}{2} \),由上即知

\[{\mathbb{E}}_{S,{S}^{\prime },{\sigma }_{j}}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{\sigma }_{j}\left( {f\left( {z}_{j}^{\prime }\right)  - f\left( {z}_{j}\right) }\right)  + \mathop{\sum }\limits_{{i \neq  j}}^{m}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right) }\right\rbrack   = {\mathbb{E}}_{S,{S}^{\prime }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {\left( {f\left( {z}_{j}^{\prime }\right)  - f\left( {z}_{j}\right) }\right)  + \mathop{\sum }\limits_{{i \neq  j}}^{m}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right) }\right\rbrack .\]

\tag{2.3}

对所有 \( j \) 使用获得(2.4)的类似处理技巧，可知

\[{\mathbb{E}}_{S,{S}^{\prime }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right\rbrack   = {\mathbb{E}}_{S,{S}^{\prime },\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right\rbrack . \tag{2.4}\]

Step 3 显然,

\[\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left\lbrack  {\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right\rbrack   \leq  \mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}f\left( {z}_{i}^{\prime }\right)  + \mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}f\left( {z}_{i}\right),\]

上式两侧关于 \( S,{S}^{\prime } \) 和 \( \mathbf{\sigma } \) 取期望,并由(2.4)和(2.2)立知

\[{\mathbb{E}}_{S}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{D}\left( f\right)  - {L}_{S}\left( f\right) }\right) }\right\rbrack\]

\[\leq  \frac{1}{m}{\mathbb{E}}_{S,{S}^{\prime },\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}\left( {f\left( {z}_{i}^{\prime }\right)  - f\left( {z}_{i}\right) }\right) }\right\rbrack\]

\[\leq  {\mathbb{E}}_{S,{S}^{\prime },\mathbf{\sigma }}\left\lbrack  {\frac{1}{m}\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}f\left( {z}_{i}^{\prime }\right)  + \frac{1}{m}\mathop{\sup }\limits_{{f \in  \mathcal{F}}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}f\left( {z}_{i}\right)}\right\rbrack\]

\[\leq  2{\mathbb{E}}_{S}\left\lbrack  {{\widehat{R}}_{S}\left( \mathcal{F}\right) }\right\rbrack .\]

结果得证.

定理 2.2 对于实值函数空间 \( \mathcal{F}: \mathcal{Z} \rightarrow  \left\lbrack  {0,1}\right\rbrack \),从分布 \( \mathcal{D} \) 中独立同分布采样得到训练集 \( Z = \; \left\{  {{z}_{1},\cdots,{z}_{m}}\right\} ,{z}_{i} \in  \mathcal{Z} \),对于任意 \( f \in  \mathcal{F} \) 和 \( 0 < \delta  < 1 \),则至少以 \( 1 - \delta \) 的概率成立以下结果

\[{L}_{\mathcal{D}}\left( f\right)  - {L}_{Z}\left( f\right)  \leq  2{R}_{m}\left( \mathcal{F}\right)  + \sqrt{\frac{\ln \left( {1/\delta }\right) }{2m}}, \tag{2.5}\]

\[{L}_{\mathcal{D}}\left( f\right)  - {L}_{Z}\left( f\right)  \leq  2{\widehat{R}}_{Z}\left( \mathcal{F}\right)  + 3\sqrt{\frac{\ln \left( {2/\delta }\right) }{2m}}. \tag{2.6}\]

证明.

Step 1 构造满足McDiarmid不等式的随机变量. 若 \( {Z}^{\prime } \) 是与 \( Z \) 仅有一个样本不同的训练集,不妨设 \( {z}_{m} \in  Z,{z}_{m}^{\prime } \in  {Z}^{\prime } \) 为不同样本. 易知

\[{\operatorname{Rep}}_{\mathcal{D}}\left( {f,{Z}^{\prime }}\right)  - {\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right)  = \mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{\mathcal{D}}\left( f\right)  - {L}_{{Z}^{\prime }}\left( f\right) }\right)  - \mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{\mathcal{D}}\left( f\right)  - {L}_{Z}\left( f\right) }\right)\]

\[\leq  \mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( {{L}_{Z}\left( f\right)  - {L}_{{Z}^{\prime }}\left( f\right) }\right)  = \mathop{\sup }\limits_{{f \in  \mathcal{F}}}\left( \frac{f\left( {z}_{m}\right)  - f\left( {z}_{m}^{\prime }\right) }{m}\right)\]

\[\leq  \frac{1}{m}\]

同理可知

\[{\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right)  - {\operatorname{Rep}}_{\mathcal{D}}\left( {f,{Z}^{\prime }}\right)  \leq  \frac{1}{m},\]

从而有

\[\left| {{\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right)  - {\operatorname{Rep}}_{\mathcal{D}}\left( {f,{Z}^{\prime }}\right) }\right|  \leq  \frac{1}{m}. \tag{2.7}\]

Step 2 证明(2.5). 对于任意 \( f \in  \mathcal{F} \),易知

\[{L}_{\mathcal{D}}\left( f\right)  - {L}_{S}\left( f\right)  \leq  {\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right).\]

而根据McDiarmid不等式,对于 \( 0 < \delta  < 1 \),

\[{\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right)  \leq  {\mathbb{E}}_{Z}\left\lbrack  {{\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right) }\right\rbrack   + \sqrt{\frac{\ln \left( {1/\delta }\right) }{2m}} \tag{2.8}\]

至少以 \( 1 - \delta \) 的概率成立,另一方面,由定理2.1知

\[{\mathbb{E}}_{Z}\left\lbrack  {{\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right) }\right\rbrack   \leq  2{\mathbb{E}}_{Z}\left\lbrack  {{\widehat{R}}_{S}\left( \mathcal{F}\right) }\right\rbrack   = 2{R}_{m}\left( \mathcal{F}\right). \tag{2.9}\]

由此可知(2.5).

Step 3 证明(2.6). 使用证明(2.7)的技巧, 类似可知

\[\left| {{\widehat{R}}_{Z}\left( \mathcal{F}\right)  - {\widehat{R}}_{{Z}^{\prime }}\left( \mathcal{F}\right) }\right|  \leq  \frac{1}{m}. \tag{2.10}\]

对于 \( 0 < \delta  < 1 \),根据McDiarmid不等式可知

\[{R}_{m}\left( \mathcal{F}\right)  = {\mathbb{E}}_{Z}\left\lbrack  {{\widehat{R}}_{Z}\left( \mathcal{F}\right) }\right\rbrack   \leq  {\widehat{R}}_{Z}\left( \mathcal{F}\right)  + \sqrt{\frac{\ln \left( {2/\delta }\right) }{2m}} \tag{2.11}\]

至少以 \( 1 - \delta /2 \) 的概率成立,而由(2.8) 知

\[{\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right)  \leq  {\mathbb{E}}_{Z}\left\lbrack  {{\operatorname{Rep}}_{\mathcal{D}}\left( {f, Z}\right) }\right\rbrack   + \sqrt{\frac{\ln \left( {2/\delta }\right) }{2m}}\]

至少以 \( 1 - \delta /2 \) 的概率成立. 联立(2.8)-(2.11)知

\[{L}_{\mathcal{D}}\left( f\right)  - {L}_{Z}\left( f\right)  \leq  2\left( {{\widehat{R}}_{Z}\left( \mathcal{F}\right)  + \sqrt{\frac{\ln \left( {2/\delta }\right) }{2m}}}\right)  + \sqrt{\frac{\ln \left( {2/\delta }\right) }{2m}}\]

至少以 \( 1 - \delta \) 的概率成立,从而 (2.6)得证.

定理 2.3 如果 \( {\phi }_{i}: \mathbb{R} \rightarrow  \mathbb{R} \) 是 \( \rho \) -Lipschitz函数,即 \( \left| {{\phi }_{i}\left( \mathbf{\alpha }\right)  - {\phi }_{i}\left( \beta \right) }\right|  \leq  \rho \left| {\mathbf{\alpha } - \beta }\right|,\forall \mathbf{\alpha },\beta  \in  \mathbb{R} \). 记 \( \mathbf{\phi }\left( \mathbf{a}\right)  = \left\lbrack  {{\phi }_{1}\left( {a}_{1}\right),\cdots,{\phi }_{i}\left( {a}_{i}\right),\cdots,{\phi }_{m}\left( {a}_{m}\right) }\right\rbrack ,\mathbf{\phi } \circ  A = \{ \mathbf{\phi }\left( \mathbf{a}\right) : \mathbf{a} \in  A\} \). 则

\[R\left( {\phi  \circ  A}\right)  \leq  {\rho R}\left( A\right). \tag{2.12}\]

证明. 不失一般性,可设 \( \rho  = 1 \).

Step 1 使用递归的技巧易知, 只需证明一个分量作变换的情形结果成立即可, 即证

\[R\left( {A}_{i}\right)  \leq  R\left( A\right) \tag{2.13}\]

式中 \( {A}_{i} = \left\{  {\left( {{a}_{1},\cdots,{a}_{i - 1},{\phi }_{i}\left( {a}_{i}\right),{a}_{i + 1},\cdots,{a}_{m}}\right) : \mathbf{a} \in  A}\right\} \)

Step 2 现证(2.13),不失一般性,记 \( i = 1 \),并以 \( \phi \) 表示 \( {\phi }_{1} \). 根据定义知

\[{mR}\left( {A}_{1}\right)  = {\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  {A}_{1}}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{a}_{i}}\right\rbrack   = {\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  A}}\left( {{\sigma }_{1}\phi \left( {a}_{1}\right)  + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i}}\right) }\right\rbrack\]

\[= \frac{1}{2}{\mathbb{E}}_{{\sigma }_{2},\cdots,{\sigma }_{m}}\left\lbrack  {\mathop{\sup }\limits_{{a \in  A}}\left( {\phi \left( {a}_{1}\right)  + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i}}\right)  + \mathop{\sup }\limits_{{a \in  A}}\left( {-\phi \left( {a}_{1}\right)  + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i}}\right) }\right\rbrack\]

\[= \frac{1}{2}{\mathbb{E}}_{{\sigma }_{2},\cdots,{\sigma }_{m}}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a},{\mathbf{a}}^{\prime } \in  A}}\left( {\phi \left( {a}_{1}\right)  - \phi \left( {a}_{1}^{\prime }\right)  + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i} + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i}^{\prime }}\right) }\right\rbrack\]

\[\leq  \frac{1}{2}{\mathbb{E}}_{{\sigma }_{2},\cdots,{\sigma }_{m}}\left\lbrack  {\mathop{\sup }\limits_{{a,{a}^{\prime } \in  A}}\left( {\left| {\phi \left( {a}_{1}\right)  - \phi \left( {a}_{1}^{\prime }\right) }\right|  + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i} + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i}^{\prime }}\right) }\right\rbrack\]

\[\leq  \frac{1}{2}{\mathbb{E}}_{{\sigma }_{2},\cdots,{\sigma }_{m}}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a},{\mathbf{a}}^{\prime } \in  A}}\left( {\left| {{a}_{1} - {a}_{1}^{\prime }}\right|  + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i} + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i}^{\prime }}\right) }\right\rbrack\]

\[= \frac{1}{2}{\mathbb{E}}_{{\sigma }_{2},\cdots,{\sigma }_{m}}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a},{\mathbf{a}}^{\prime } \in  A}}\left( {{a}_{1} - {a}_{1}^{\prime } + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i} + \mathop{\sum }\limits_{{i = 2}}^{m}{\sigma }_{i}{a}_{i}^{\prime }}\right) }\right\rbrack\]

\[= {mR}\left( A\right) \text{. }\]

由此证得结果.

引理 2.1 对任一 \( A \subset  {\mathbb{R}}^{m}, c \in  \mathbb{R},{\mathbf{a}}_{\mathbf{0}} \in  {\mathbb{R}}^{m} \),有

\[R\left( \left\{  {c\mathbf{a} + {\mathbf{a}}_{0}: \mathbf{a} \in  A}\right\}  \right)  \leq  \left| c\right| R\left( A\right). \tag{2.14}\]

证明. 记 \( \mathbf{b} = {\mathbf{a}}_{0} \),由定义知

\[{mR}\left( {\{ c\mathbf{a} + {\mathbf{a}}_{0}: \mathbf{a} \in  A\} }\right)  = {\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  A}}\left( {\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}\left( {c{a}_{i} + {b}_{i}}\right) }\right) }\right\rbrack\]

\[= {\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  A}}\left( {c\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{a}_{i}}\right)  + \mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{b}_{i}}\right\rbrack\]

\[= c{\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  A}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{a}_{i}}\right\rbrack   \leq  \left| c\right| {mR}\left( A\right).\]

由此证得结果.

定理 2.4 (The Massart lemma) 设 \( A = \left\lbrack  {{\mathbf{a}}_{1},\cdots,{\mathbf{a}}_{N}}\right\rbrack   \subset  {\mathbb{R}}^{m} \),记 \( \overline{\mathbf{a}} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\mathbf{a}}_{i} \),则

\[R\left( A\right)  \leq  \mathop{\max }\limits_{{\mathbf{a} \in  A}}\parallel \mathbf{a} - \overline{\mathbf{a}}\parallel \frac{\sqrt{2\ln \left( N\right) }}{m}.\]

证明. 根据引理2.1,不失一般性,可设 \( \overline{\mathbf{a}} = \mathbf{0} \). 设 \( \lambda  > 0 \),并记 \( {A}^{\prime } = \left\lbrack  {\lambda {\mathbf{a}}_{1},\cdots,\lambda {\mathbf{a}}_{N}}\right\rbrack \),则

\[{mR}\left( {A}^{\prime }\right)  = {\mathbb{E}}_{\sigma }\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  {A}^{\prime }}}{\mathbf{\sigma }}^{T}\mathbf{a}}\right\rbrack   = {\mathbb{E}}_{\sigma }\left\lbrack  {\ln \left( {\mathop{\max }\limits_{{\mathbf{a} \in  {A}^{\prime }}}{e}^{\langle \mathbf{\sigma },\mathbf{a}\rangle }}\right) }\right\rbrack\]

\[\leq  {\mathbb{E}}_{\sigma }\left\lbrack  {\ln \left( {\mathop{\sum }\limits_{{\mathbf{a} \in  {A}^{\prime }}}{e}^{\langle \mathbf{\sigma },\mathbf{a}\rangle }}\right) }\right\rbrack\]

\[\leq  \ln \left( {{\mathbb{E}}_{\mathbf{\sigma }}\left\lbrack  {\mathop{\sum }\limits_{{\mathbf{a} \in  {A}^{\prime }}}{e}^{\langle \mathbf{\sigma },\mathbf{a}\rangle }}\right\rbrack  }\right)\]

\[= \ln \left( {\mathop{\sum }\limits_{{\mathbf{a} \in  {A}^{\prime }}}\mathop{\prod }\limits_{{i = 1}}^{m}{\mathbb{E}}_{{\sigma }_{i}}\left\lbrack  {e}^{{\sigma }_{i}{a}_{i}}\right\rbrack  }\right),\]

而

\[{\mathbb{E}}_{{\sigma }_{i}}\left\lbrack  {e}^{{\sigma }_{i}{a}_{i}}\right\rbrack   = \frac{1}{2}\left( {{e}^{{a}_{i}} + {e}^{-{a}_{i}}}\right)  \leq  {e}^{{a}_{i}^{2}/2},\]

因此

\[{mR}\left( {A}^{\prime }\right)  \leq  \ln \left( {\mathop{\sum }\limits_{{\mathbf{a} \in  {A}^{\prime }}}\mathop{\prod }\limits_{{i = 1}}^{m}{e}^{{a}_{i}^{2}/2}}\right)  = \ln \left( {\mathop{\sum }\limits_{{\mathbf{a} \in  {A}^{\prime }}}{e}^{\parallel \mathbf{a}{\parallel }^{2}/2}}\right)\]

\[\leq  \ln \left( {N\mathop{\max }\limits_{{\mathbf{a} \in  {A}^{\prime }}}{e}^{\parallel \mathbf{a}{\parallel }^{2}/2}}\right)  = \ln \left( N\right)  + \frac{1}{2}\mathop{\max }\limits_{{\mathbf{a} \in  {A}^{\prime }}}\parallel \mathbf{a}{\parallel }^{2}.\]

而 \( R\left( {A}^{\prime }\right)  = \frac{1}{\lambda }R\left( A\right) \),故知

\[R\left( A\right)  \leq  \frac{\ln \left( N\right)  + \frac{1}{2}{\lambda }^{2}\mathop{\max }\limits_{{\mathbf{a} \in  A}}\parallel \mathbf{a}{\parallel }^{2}}{\lambda m}\]

特取 \( \lambda  = \sqrt{2\ln \left( N\right) /\mathop{\max }\limits_{{\mathbf{a} \in  A}}\parallel \mathbf{a}{\parallel }^{2}} \) 即得所需结果.

## 3 VC 维数

令 \( \mathcal{H} \) 为假设空间,其中的假设是 \( \mathcal{X} \rightarrow  \mathcal{Y} = \{  - 1,1\} \) 的映射,对于数据集 \( D = \left\{  {{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{m}}\right\}   \subset \; \mathcal{X},\mathcal{H} \) 在数据集 \( D \) 上的限制是从 \( D \) 到 \( \{  - 1,1\}^{m} \) 的一族映射:

\[{\left. \mathcal{H}\right| }_{D} = \left\{  {\left( {h\left( {\mathbf{x}}_{1}\right),\cdots, h\left( {\mathbf{x}}_{m}\right) }\right)  \mid  h \in  \mathcal{H}}\right\} ,\]

其中 \( h \) 在 \( D \) 上的限制是一个 \( m \) 维向量.

对于 \( m \in  \mathbb{N} \),假设空间 \( \mathcal{H} \) 的增长函数 \( {\Pi }_{\mathcal{H}}\left( m\right) \) 定义为

\[{\Pi }_{\mathcal{H}}\left( m\right)  = \mathop{\max }\limits_{{\left\{  {{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{m}}\right\}   \subset  \mathcal{X}}}\left| \left\{  {\left( {h\left( {\mathbf{x}}_{1}\right),\cdots, h\left( {\mathbf{x}}_{m}\right) }\right)  \mid  h \in  \mathcal{H}}\right\}  \right|,\]

对于大小为 \( m \) 的数据集 \( D \),有

\[{\Pi }_{\mathcal{H}}\left( m\right)  = \mathop{\max }\limits_{{\left| D\right|  = m}}\left| {\left.\mathcal{H}\right|}_{D}\right|\]

增长函数 \( {\Pi }_{\mathcal{H}}\left( m\right) \) 描述了 \( h \in  \mathcal{H} \) 时， \( \left( {h\left( {\mathbf{x}}_{1}\right),\cdots, h\left( {\mathbf{x}}_{m}\right) }\right) \) 为相异点的个数.

假设空间 \( \mathcal{H} \) 中不同的假设对于 \( D \) 中的样本赋予标记的结果可能相同，也可能不同. 例如， 对于二分类问题，对 \( m \) 个样本最多有 \( {2}^{m} \) 个可能的结果.

对于二分类问题，假设空间 \( \mathcal{H} \) 中的假设对 \( D \) 中的样本赋予标记的每种可能结果称为对 \( D \) 的一种对分. 如果假设空间 \( \mathcal{H} \) 能实现样本集 \( D \) 上的所有对分,即 \( \left| {\left.\mathcal{H}\right|}_{D}\right|  = {2}^{m} \),则称样本集 \( D \) 能被假设空间 \( \mathcal{H} \) 打散,此时 \( {\Pi }_{\mathcal{H}}\left( m\right)  = {2}^{m} \).

定义 3.1 假设空间 \( \mathcal{H} \) 的 \( {VC} \) 维是能被 \( \mathcal{H} \) 打散的最大样本集的大小,即

\[\operatorname{VC}\left( \mathcal{H}\right)  = \max \left\{  {m: {\Pi }_{\mathcal{H}}\left( m\right)  = {2}^{m}}\right\} .\]

若 \( \operatorname{VC}\left( \mathcal{H}\right)  = d \),则对于 \( \left( {{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{d}}\right) \) 的任一状态,都存在 \( h \in  \mathcal{H} \) 实现该状态.

引理 3.1 若假设空间的 \( {VC} \) 维为 \( d \),则对任意 \( m \in  \mathbb{N} \),

\[{\Pi }_{\mathcal{H}}\left( m\right)  \leq  \mathop{\sum }\limits_{{i = 0}}^{d}{C}_{m}^{i}\]

定理 3.1 若假设空间的 \( {VC} \) 维为 \( d \),则对任意自然数 \( m \geq  d \) 有

\[{\Pi }_{\mathcal{H}}\left( m\right)  \leq  {\left( \frac{e \cdot  m}{d}\right) }^{d}.\]

需要指出的是, VC 维是针对二分类问题定义的. 对于多分类问题, 也可以有相应的假设空间复杂度刻画方法,即Natarajan维. 在多分类问题中,假设空间 \( \mathcal{H} \) 的假设是 \( \mathcal{X} \rightarrow  \mathcal{Y} = \; \{ 0,\cdots, K - 1\} \) 的映射,其中 \( K \) 为类别数. 类似于二分类问题,可定义增长函数与打散.

定义 3.2 对于多分类问题的假设空间 \( \mathcal{H} \),其 Natarajan 维Natarajan \( \left( \mathcal{H}\right) \) 是能被 \( \mathcal{H} \) 打散的最大样本集的大小.

定理 3.2 当类别数 \( K = 2 \) 时, \( \operatorname{VC}\left( \mathcal{H}\right)  = \operatorname{Natarajan}\left( \mathcal{H}\right) \).

定义 3.3 若多分类问题的假设空间 \( \mathcal{H} \) 的Natarajan维为 \( d \),类别数为 \( K \),则对任意 \( m \in  \mathbb{N} \),

\[{\Pi }_{\mathcal{H}}\left( m\right)  \leq  {m}^{d}{K}^{2d}.\]

定理 3.3 假设空间 \( \mathcal{H} \) 的Rademacher复杂度 \( {R}_{m}\left( \mathcal{H}\right) \) 与增长函数 \( {\Pi }_{\mathcal{H}}\left( m\right) \) 之间满足

\[{R}_{m}\left( \mathcal{H}\right)  \leq  \sqrt{\frac{2\ln \left( {{\Pi }_{\mathcal{H}}\left( m\right) }\right) }{m}}. \tag{3.1}\]

证明. 对于 \( D = \left\{  {{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{m}}\right\} ,{\left. \mathcal{H}\right| }_{D} \) 是假设空间 \( \mathcal{H} \) 在 \( D \) 上的限制. 由于 \( h \in  \mathcal{H} \) 的值域为 \( \{  - 1,1\} \),可知 \( {\left. \mathcal{H}\right| }_{D} \) 中元素为模长 \( \sqrt{m} \) 的向量. 因此,由定理 2.4 知

\[{R}_{m}\left( \mathcal{H}\right)  = {\mathbb{E}}_{D}\left\lbrack  {{\mathbb{E}}_{\sigma }\left\lbrack  {\mathop{\sup }\limits_{{u \in  \mathcal{H}{|}_{D}}}\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{u}_{i}}\right\rbrack  }\right\rbrack   \leq  {\mathbb{E}}_{D}\left\lbrack  {\sqrt{m}\frac{\sqrt{2\ln \left( {\left| \mathcal{H}\right| }_{D}\right) }}{m}}\right\rbrack .\]

又因为 \( \left| {\left.\mathcal{H}\right|}_{D}\right|   \leq  {\Pi }_{\mathcal{H}}\left( m\right) \),故知

\[{R}_{m}\left( \mathcal{H}\right)  \leq  {\mathbb{E}}_{D}\left\lbrack  \frac{\sqrt{2\ln \left( {{\Pi }_{\mathcal{H}}\left( m\right) }\right) }}{\sqrt{m}}\right\rbrack   = \sqrt{\frac{2\ln \left( {{\Pi }_{\mathcal{H}}\left( m\right) }\right) }{m}}.\]

■

需要指出的是VC维与数据分布无关，而Rademacher复杂度依赖于具体问题及数据分布， 定理3.3明确了Rademacher复杂度与增长函数的关系.

## 4 求解微分方程的二层神经网络算法误差分析

在本节中, 参考文献 [4], 给出求解微分方程的单隐层人工智能算法的误差分析. 考虑如下模型问题:

\[\begin{cases}  - {\Delta u} + u &  = f\text{ in }\Omega, \\  {\partial }_{\mathbf{n}}u &  = 0\text{ on }\partial \Omega, \end{cases} \tag{4.1}\]

式中 \( \Omega  \subset  {\mathbb{R}}^{d} \) 是一个具边界 \( \partial \Omega \) 的充分光滑区域. 该问题可等价描述为如下优化问题:

\[u = \underset{v \in  {H}^{1}\left( \Omega \right) }{\arg \min }J\left( v\right)  = \frac{1}{2}a\left( {v, v}\right)  - {\int }_{\Omega }{fv}\mathrm{\; d}x, \tag{4.2}\]

式中,对任意 \( v, w \in  {H}^{1}\left( \Omega \right) \),

\[a\left( {v, w}\right)  = {\int }_{\Omega }\left( {\nabla v \cdot  \nabla w + {vw}}\right) \mathrm{d}x.\]

对于具单隐层的二层神经网络, 其输出的函数是如下形式函数的线性组合:

\[\mathbb{D} = {\mathbb{D}}_{\sigma } = \{ \sigma \left( {\mathbf{w} \cdot  \mathbf{x} + b}\right) : \left( {\mathbf{w}, b}\right)  \in  \Theta \}  \subset  {H}^{1}\left( \Omega \right),\]

式中,参数集 \( \mathbf{\Theta } \subset  {\mathbb{R}}^{d + 1} \) 取为紧集. 称 \( \mathbb{D} \) 为字典集 (Dictionary set).

对于任一自然数 \( n \) 和正实数 \( M \),定义

\[{\sum }_{n, M}\left( \mathbb{D}\right)  = \left\{  {\mathop{\sum }\limits_{{i = 1}}^{n}{a}_{i}{d}_{i}: {a}_{i} \in  \mathbb{R},{d}_{i} \in  \mathbb{D},\mathop{\sum }\limits_{{i = 1}}^{n}\left| {a}_{i}\right|  \leq  M}\right\} .\]

又记

\[{B}_{M}\left( \mathbb{D}\right)  = \overline{\mathop{\bigcup }\limits_{{n = 1}}^{\infty }{\sum }_{n, M}\left( \mathbb{D}\right) },\]

式中的完备化是在 \( {H}^{1}\left( \Omega \right) \) 的度量下进行的. 可以证明

\[{B}_{M}\left( \mathbb{D}\right)  = \left\{  {u \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) : \parallel u{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M}\right\} ,\]

式中 \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) 为常规的Barron空间.

构造如下参数化优化问题:

\[{u}_{\theta } = \underset{v \in  {B}_{M}\left( \mathbb{D}\right) }{\arg \min }J\left( v\right) \tag{4.3}\]

\[{u}_{\theta, N} = \underset{v \in  {B}_{M}\left( \mathbb{D}\right) }{\arg \min }{J}_{N}\left( v\right) \tag{4.4}\]

式中

\[{J}_{N}\left( v\right)  = \frac{1}{2}\left( {\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\left| \nabla v\right| }^{2}\left( {\mathbf{x}}_{i}\right)  + {v}^{2}\left( {\mathbf{x}}_{i}\right)  - \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}f\left( {\mathbf{x}}_{i}\right) v\left( {\mathbf{x}}_{i}\right) }\right),\]

而 \( {\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N} \) 是基于 \( \Omega \) 的一致分布所取的独立同分布样本点. 易知,(4.4)实即求解问题(4.2)的单隐层人工智能算法.

考虑到 \( u \) 为问题 (4.2) 的最优解,直接计算易知

\[J\left( {u}_{\theta, N}\right)  - J\left( u\right)  = {J}^{\prime }\left( u\right) \left( {{u}_{\theta, N} - u}\right)  + \frac{1}{2}{J}^{\prime \prime }\left( u\right) \left( {{u}_{\theta, N} - u,{u}_{\theta, N} - u}\right)  = {\begin{Vmatrix}{u}_{\theta, N} - u\end{Vmatrix}}_{1,\Omega }^{2}. \tag{4.5}\]

于是为得到误差估计,关键是估计上式左端项. 而注意到 \( {J}_{N}\left( {u}_{\theta, N}\right)  \leq  {J}_{N}\left( {u}_{\theta }\right) \),可知

\[J\left( {u}_{\theta, N}\right)  - J\left( u\right)\]

\[\leq  J\left( {u}_{\theta, N}\right)  - {J}_{N}\left( {u}_{\theta, N}\right)  + {J}_{N}\left( {u}_{\theta, N}\right)  - {J}_{N}\left( {u}_{\theta }\right)  + {J}_{N}\left( {u}_{\theta }\right)  - J\left( {u}_{\theta }\right)  + J\left( {u}_{\theta }\right)  - J\left( u\right) \tag{4.6}\]

\[\leq  J\left( {u}_{\theta }\right)  - J\left( u\right)  + 2\mathop{\sup }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}\left| {J\left( v\right)  - {J}_{N}\left( v\right) }\right|.\]

前一项的估计涉及神经网络函数对解 \( u \) 的逼近能力的刻划,已有很多结果,下面我们重点估计第二项, 它可理解为由Monte Carlo逼近积分带来的误差, 此时要使用Rademacher复杂度及理论来获得估计. 在后文推导中,给定两个量 \( a \) 和 \( b \),用 “ \( a \lesssim  b \) ” 表示 “ \( a \leq  {Cb} \) ”,式中一般常数 \( C \) 在不同地方出现可取不同之值,但均不依赖于Monte Carlo抽样点数 \( N \) 或隐层神经单元个数 \( n \) 等参数.

首先, 由定理2.1立知

定理 4.1 设 \( \mathcal{F} \) 为一函数集,则成立

\[{\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N} \sim  \mu }\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\left| {\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}h\left( {\mathbf{x}}_{i}\right) -\int h\left( \mathbf{x}\right) {d\mu }}\right|  \leq  2{R}_{N}\left( \mathcal{F}\right),\]

式中 \( \mu \) 是 \( \Omega \) 上的一致分布,而 \( {R}_{N}\left( \mathcal{F}\right) \) 为 Rademacher 复杂度,

\[{R}_{N}\left( \mathcal{F}\right)  = {\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N} \sim  \mu }{\mathbb{E}}_{{\xi }_{1},\cdots,{\xi }_{N}}\left\lbrack  {\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\xi }_{i}h\left( {\mathbf{x}}_{i}\right) }\right\rbrack .\]

此处, \( {\xi }_{1},\ldots,{\xi }_{n} \) 是独立同分布的 Rademacher 变量,即为等概率取 \( \pm  1 \) 的随机变量.

以下结果给出了Rademacher复杂度的基本性质, 部分在第二节中给予了证明.

引理 4.1 设 \( \mathcal{F},\mathcal{S} \) 是定义在 \( \Omega \) 的两个函数类,则以下结果成立:

1. \( {R}_{N}\left( {\operatorname{conv}\left( \mathcal{F}\right) }\right)  = {R}_{N}\left( \mathcal{F}\right) \)

2. 定义 \( \mathcal{F} + \mathcal{S} = \{ h\left( \mathbf{x}\right)  + g\left( \mathbf{x}\right) : h \in  \mathcal{F}, g \in  \mathcal{S}\} \). 则

\[{R}_{N}\left( {\mathcal{F} + \mathcal{S}}\right)  = {R}_{N}\left( \mathcal{F}\right)  + {R}_{N}\left( \mathcal{S}\right).\]

3. 若 \( \phi : \mathbb{R} \rightarrow  \mathbb{R} \) 是 \( L \) -Lipschitz连续的,则 \( \phi  \circ  \mathcal{F} = \{ \phi \left( {h\left( \mathbf{x}\right) }\right) : h \in  \mathcal{F}\} \) 成立估计

\[{R}_{N}\left( {\phi  \circ  \mathcal{F}}\right)  \leq  L{R}_{N}\left( \mathcal{F}\right).\]

4. 若 \( f: \Omega  \rightarrow  \mathbb{R} \) 是一个给定函数,则 \( f \cdot  \mathcal{F} = \{ f\left( \mathbf{x}\right) h\left( \mathbf{x}\right) : h \in  \mathcal{F}\} \) 成立估计

\[{R}_{N}\left( {f \cdot  \mathcal{F}}\right)  \leq  \parallel f{\parallel }_{0,\infty }{R}_{N}\left( \mathcal{F}\right).\]

在以下分析中, 特取函数类为

\[{\mathcal{F}}_{n, M} = \left\{  {\frac{1}{2}\left( {{\left| \nabla v\left( \mathbf{x}\right) \right| }^{2} + \frac{1}{2}v{\left( \mathbf{x}\right) }^{2} - f\left( \mathbf{x}\right) v\left( \mathbf{x}\right) }\right) : v \in  {\sum }_{n, M}\left( \mathbb{D}\right) }\right\} .\]

引理 4.2 若 \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{2,\infty } \lesssim  1 \),则

\[{R}_{N}\left( {\mathcal{F}}_{n, M}\right)  \lesssim  {M}^{2}\mathop{\sum }\limits_{{\left| \mathbf{\alpha }\right|  = 1}}{R}_{N}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right)  + {M}^{2}{R}_{N}\left( \mathbb{D}\right)  + \parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) }M{R}_{N}\left( \mathbb{D}\right),\]

式中, \( {\partial }^{\mathbf{\alpha }}\mathbb{D} = \left\{  {{\partial }^{\mathbf{\alpha }}d: d \in  \mathbb{D}}\right\} \).

证明. 记 \( \phi \left( x\right)  = \frac{1}{2}{x}^{2} \). 显然,

\[{\mathcal{F}}_{n, M} \subset  \mathop{\sum }\limits_{{\left| \mathbf{\alpha }\right|  = 1}}\left( {\phi  \circ  {\sum }_{n, M}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right) }\right)  + \phi  \circ  {\sum }_{n, M}\left( \mathbb{D}\right)  + f \cdot  {\sum }_{n, M}\left( \mathbb{D}\right).\]

对任意 \( {\partial }^{\mathbf{\alpha }}v \in  {\sum }_{n, M}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right) \),式中 \( v \in  {\sum }_{n, M}\left( \mathbb{D}\right),\left| \mathbf{\alpha }\right|  \leq  2 \),成立

\[\left| {{\partial }^{\mathbf{\alpha }}v}\right|  = \left| {{\sum }_{i = 1}^{n}{a}_{i}{\partial }^{\mathbf{\alpha }}{h}_{i}}\right|  \leq  {\sum }_{i = 1}^{n}\left| {a}_{i}\right| \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{2,\infty } \lesssim  M\]

\[\left| v\right|  = \left| {{\sum }_{i = 1}^{n}{a}_{i}{h}_{i}}\right|  \leq  {\sum }_{i = 1}^{n}\left| {a}_{i}\right| \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{0,\infty } \lesssim  M.\]

因此, \( \phi \) 关于 \( {\sum }_{n, M}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right) \) 和 \( {\sum }_{n, M}\left( \mathbb{D}\right) \) 中的函数是Lipschitz连续的. 于是根据引理4.1可得

\[\begin{aligned}
{R}_{N}\left( {\mathcal{F}}_{n, M}\right)  &\lesssim  M{R}_{N}\left( {{\sum }_{n, M}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right) }\right)  + M{R}_{N}\left( {{\sum }_{n, M}\left( \mathbb{D}\right) }\right)  + \parallel f{\parallel }_{0,\infty }{R}_{N}\left( {{\sum }_{n, M}\left( \mathbb{D}\right) }\right) \\
&\lesssim  {M}^{2}{R}_{N}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right)  + {M}^{2}{R}_{N}\left( \mathbb{D}\right)  + \parallel f{\parallel }_{0,\infty }M{R}_{N}\left( \mathbb{D}\right).
\end{aligned} \tag{4.7}\]

■

定理 4.2 若 \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{2,\infty } \lesssim  1 \),则成立

\[{\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N}}\mathop{\sup }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}\left| {{J}_{N}\left( v\right)  - J\left( v\right) }\right|  \lesssim  2{M}^{2}\mathop{\sum }\limits_{{\left| \mathbf{\alpha }\right|  = 1}}{R}_{N}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right)  + 2{M}^{2}{R}_{N}\left( \mathbb{D}\right)  + 2\parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) }M{R}_{N}\left( \mathbb{D}\right). \tag{4.8}\]

证明. 由定理4.1和引理4.2, 可得

\[\begin{aligned}
{\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N}}\mathop{\sup }\limits_{{v \in  {\sum }_{n, M}\left( \mathbb{D}\right) }}\left| {{J}_{N}\left( v\right)  - J\left( v\right) }\right|  &\lesssim  2{M}^{2}\mathop{\sum }\limits_{{\left| \mathbf{\alpha }\right|  = 1}}{R}_{N}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right)  + 2{M}^{2}{R}_{N}\left( \mathbb{D}\right) \\
&\quad + 2\parallel f{\parallel }_{0,\infty }M{R}_{N}\left( \mathbb{D}\right) \text{. }
\end{aligned} \tag{4.9}\]

对任一 \( \varepsilon  > 0 \),存在 \( w \in  {B}_{M}\left( \mathbb{D}\right) \),使得

\[\mathop{\sup }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}\left| {{J}_{N}\left( v\right)  - J\left( v\right) }\right|  < \left| {{J}_{N}\left( w\right)  - J\left( w\right) }\right|  + \varepsilon.\]

又对任一 \( k \in  \mathbb{N} \),存在 \( {w}_{k} \in  {\sum }_{k, M}\left( \mathbb{D}\right) \),使得 \( {\begin{Vmatrix}{w}_{k} - w\end{Vmatrix}}_{M} \leq  \frac{1}{k} \). 故得

\[{\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N}}\mathop{\sup }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}\left| {{J}_{N}\left( v\right)  - J\left( v\right) }\right|  < {\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N}}\left| {{J}_{N}\left( w\right)  - J\left( w\right) }\right|  + \varepsilon\]

\[\leq  {\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N}}\left( {\left| {{J}_{N}\left( w\right)  - {J}_{N}\left( {w}_{k}\right) }\right|  + \left| {{J}_{N}\left( {w}_{k}\right)  - J\left( {w}_{k}\right) }\right| }\right.\]

\[+ \left. \left| {J\left( {w}_{k}\right)  - J\left( w\right) }\right| \right)  + \varepsilon.\]

于是由(4.9)可知

\[{\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N}}\mathop{\sup }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}\left| {{J}_{N}\left( v\right)  - J\left( v\right) }\right|  \lesssim  \frac{2}{k} + \varepsilon  + {\mathbb{E}}_{{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{N}}\left| {{J}_{N}\left( {w}_{k}\right)  - J\left( {w}_{k}\right) }\right|\]

\[\lesssim  \frac{2}{k} + \varepsilon  + 2{M}^{2}\mathop{\sum }\limits_{{\left| \mathbf{\alpha }\right|  = 1}}{R}_{N}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right)\]

\[+ 2{M}^{2}{R}_{N}\left( \mathbb{D}\right)  + 2\parallel f{\parallel }_{0,\infty }M{R}_{N}\left( \mathbb{D}\right)\]

再令 \( k \rightarrow  \infty \) 和 \( \varepsilon  \rightarrow  0 + \) 即得结果.

以下结果取自 [1]:

引理 4.3 设 \( S = \left\lbrack  {{\mathbf{x}}_{1},\cdots,{\mathbf{x}}_{m}}\right\rbrack \) 是某一 Hilbert 空间的 \( m \) 个向量. 定义

\[{\mathcal{H}}_{2} \circ  S = \left\{  {\left( {\left\langle  {\mathbf{w},{\mathbf{x}}_{1}}\right\rangle ,\cdots,\left\langle  {\mathbf{w},{\mathbf{x}}_{m}}\right\rangle  }\right) : \parallel \mathbf{w}{\parallel }_{2} \leq  1}\right\} .\]

则成立

\[R\left( {{\mathcal{H}}_{2} \circ  S}\right)  \leq  \frac{\mathop{\max }\limits_{i}{\begin{Vmatrix}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}}{\sqrt{m}}.\]

证明. 使用Cauchy-Schwarz不等式可得

\[{mR}\left( {{\mathcal{H}}_{2} \circ  S}\right)  = {\mathbb{E}}_{\sigma }\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{a} \in  {\mathcal{H}}_{2} \circ  S}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{a}_{i}}\right\rbrack\]

\[= {\mathbb{E}}_{\sigma }\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{w}: \parallel \mathbf{w}\parallel  \leq  1}}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}\left\langle  {\mathbf{w},{\mathbf{x}}_{i}}\right\rangle  }\right\rbrack\]

\[= {\mathbb{E}}_{\sigma }\left\lbrack  {\mathop{\sup }\limits_{{\mathbf{w}: \parallel \mathbf{w}\parallel  \leq  1}}\left\langle  {\mathbf{w},\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{\mathbf{x}}_{i}}\right\rangle  }\right\rbrack\]

\[\leq  {\mathbb{E}}_{\sigma }\left\lbrack  {\begin{Vmatrix}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}\right\rbrack . \tag{4.10}\]

而由Jensen不等式有

\[\underset{\sigma }{\mathbb{E}}\left\lbrack  {\begin{Vmatrix}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}\right\rbrack   = \underset{\sigma }{\mathbb{E}}\left\lbrack  {\left( {\begin{Vmatrix}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}^{2}\right) }^{1/2}\right\rbrack   \leq  {\left( \underset{\sigma }{\mathbb{E}}\left\lbrack  {\begin{Vmatrix}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}^{2}\right\rbrack  \right) }^{1/2}. \tag{4.11}\]

考虑到随机变量 \( {\sigma }_{1},\ldots,{\sigma }_{m} \) 是独立同分布的,可知

\[{\mathbb{E}}_{\sigma }\left\lbrack  {\begin{Vmatrix}\mathop{\sum }\limits_{{i = 1}}^{m}{\sigma }_{i}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}^{2}\right\rbrack   = {\mathbb{E}}_{\sigma }\left\lbrack  {\mathop{\sum }\limits_{{i, j}}{\sigma }_{i}{\sigma }_{j}\left\langle  {{\mathbf{x}}_{i},{\mathbf{x}}_{j}}\right\rangle  }\right\rbrack\]

\[= \mathop{\sum }\limits_{{i \neq  j}}\left\langle  {{\mathbf{x}}_{i},{\mathbf{x}}_{j}}\right\rangle  {\mathbb{E}}_{\sigma }\left\lbrack  {{\sigma }_{i}{\sigma }_{j}}\right\rbrack   + \mathop{\sum }\limits_{{i = 1}}^{m}\left\langle  {{\mathbf{x}}_{i},{\mathbf{x}}_{i}}\right\rangle  {\mathbb{E}}_{\sigma }\left\lbrack  {\sigma }_{i}^{2}\right\rbrack\]

\[= \mathop{\sum }\limits_{{i = 1}}^{m}{\begin{Vmatrix}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}^{2} \leq  m\mathop{\max }\limits_{i}{\begin{Vmatrix}{\mathbf{x}}_{i}\end{Vmatrix}}_{2}^{2}. \tag{4.12}\]

因此, 联立(4.10)-(4.12)立得欲证结果.

定理 4.3 若激活函数 \( \sigma  \in  {W}^{2,\infty } \),则对任一 \( \left| \mathbf{\alpha }\right|  \leq  1 \),成立

\[{R}_{N}\left( {{\partial }^{\mathbf{\alpha }}\left( \mathbb{D}\right) }\right)  \lesssim  {N}^{-\frac{1}{2}}.\]

证明. 显然,

\[{\partial }^{\mathbf{\alpha }}\mathbb{D} = \left\{  {{\mathbf{w}}^{\mathbf{\alpha }}{\sigma }^{\left( \mathbf{\alpha }\right) }\left( {\mathbf{w} \cdot  \mathbf{x} + b}\right) : \left( {\mathbf{w}, b}\right)  \in  \Theta }\right\} .\]

因 \( \Theta \) 为紧集,立知 \( \left| {\mathbf{w}}^{\mathbf{\alpha }}\right| \) 有界. 又因 \( \sigma  \in  {W}^{2,\infty } \),故 \( {\sigma }^{\left( \mathbf{\alpha }\right) } \) 是Lipschitz函数. 于是由引理4.1可得

\[{R}_{N}\left( {{\partial }^{\mathbf{\alpha }}\mathbb{D}}\right)  \lesssim  {R}_{N}\left( {\{ \mathbf{w} \cdot  \mathbf{x} + b: \left( {\mathbf{w}, b}\right)  \in  \Theta \} }\right).\]

而由引理4.3可知

\[{R}_{N}\left( {\{ \mathbf{w} \cdot  \mathbf{x} + b: \left( {\mathbf{w}, b}\right)  \in  \Theta \} }\right)  \lesssim  {N}^{-\frac{1}{2}}.\]

联立以上两式可得结果. ■

## 参考文献

[1] S. S. Shai and B. D. Shai, Understanding Machine Learning: From Theorem to Algorithms, Cambridge University Press, Cambridge, 2014.

[2] 周志华, 机器学习, 清华大学出版社, 北京, 2016.

[3] 周志华, 王魏, 高尉, 张利军, 机器学习理论导引, 机械工业出版社, 北京, 2020.

[4] J. W. Siegel, Q. Hong, X. Jin, W. Hao and J. Xu, Greedy training algorithms for neural networks and applications to PDEs, JCP, 484 (2023), 112084.
