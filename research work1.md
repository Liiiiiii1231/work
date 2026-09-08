# 单套 ToMA 共锚点全双工近场 ISAC 系统建模与低复杂度联合优化算法

> **公式显示说明：** 本文所有独立公式统一采用 `$$ ... $$`，以兼容常见 Markdown、Typora、Obsidian 和 VS Code 数学公式渲染器。

> **模型改动说明。** 本模型在原“单套 ToMA 共锚点全双工近场 ISAC”框架上作三项核心修改：  
> 1. 将原先多个、但只通过总和起作用的感知协方差 $\{\mathbf S_l\}$ 合并为总感知协方差，并进一步写成多波束因子形式 $\mathbf S=\mathbf V\mathbf V^H$；  
> 2. 将内层的 SCA-SDR/SDP 替换为拉格朗日对偶变换（LDT）、二次变换（QT）与总功率拉格朗日乘子相结合的分式规划算法；  
> 3. 完全删除残余自干扰功率阈值、显式 SI 约束、候选解 SI 检查和 SI 缩放初始化；同时取消残余 SI 的前端—基带分层建模，将被动隔离、模拟消除和数字消除的综合效果统一表示为等效残余自干扰信道 $\mathbf H_{\rm RSI}=\sqrt{\rho_{\rm SI}}\mathbf H_{\rm SI}^{0}$。该信道仅进入上行与感知 SINR 分母，不构成任何可行性条件或验收条件；目标回波与残余 SI 的二阶干扰统计采用互不相关的协方差累加模型，不保留 echo--echo 或 echo--RSI 交叉项。
>
> 因此，联合优化问题只保留收发功率、接收波束单位范数、缆绳定长和子无人机防碰撞约束；内层 FP 闭式更新与外层黎曼位置优化的主体结构保持不变，同时省去所有与 SI 阈值有关的回溯和检测步骤。

---

## 1. 系统假设

### 假设 A1：主无人机固定于坐标原点

主无人机作为全双工 ISAC 中心平台，其位置固定为

$$
\mathbf q_0=[0,0,0]^T.
$$

本文不优化主无人机轨迹、平移和姿态，只优化拖曳子无人机相对于主无人机的三维端点位置。

### 假设 A2：全部缆绳共用一个物理锚点

全部发射和接收缆绳连接到主无人机上的同一锚点，并令

$$
\mathbf o=\mathbf q_0=\mathbf 0.
$$

共锚点仅表示缆绳具有相同的几何起点，不表示发射阵元与接收阵元物理重合。

### 假设 A3：固定的收发子无人机分组

子无人机总数 $M$ 为偶数，其中 $M/2$ 架固定拖曳发射子阵列，另外 $M/2$ 架固定拖曳接收子阵列。收发功能在一个优化周期内不切换，因此不引入二进制分组变量。

### 假设 A4：确定性窄带纯 LoS 球面波信道

下行、上行和目标回波采用均匀球面波（uniform spherical wave, USW）形式：链路的公共幅度衰减归入复路径增益，阵列响应保留逐阵元精确传播距离产生的球面波相位。共锚点近距离直接 SI 使用逐收发阵元对的自由空间幅度衰减和球面波相位。

本文暂不考虑 NLoS 多径、平台抖动、缆绳弯曲、宽带波束偏斜、互耦和方向图畸变。

### 假设 A5：忽略 UL 用户到 DL 用户的交叉链路干扰

在下行用户接收信号中忽略上行用户造成的交叉链路干扰。该假设适用于用户间距离较远、上行功率较低、用户采用高定向天线，或系统已采用调度与交叉链路干扰消除的场景。

### 假设 A6：感知信号采用多波束因子形式

总感知发射信号由 $R_{\rm S}$ 个相互独立的感知波形承载：

$$
\mathbf s=\mathbf V\mathbf z
=\sum_{r=1}^{R_{\rm S}}\mathbf v_r z_r,
$$

其中

$$
\mathbf V=[\mathbf v_1,\ldots,\mathbf v_{R_{\rm S}}]
\in\mathbb C^{N_{\rm T}\times R_{\rm S}},
\qquad
\mathbb E[\mathbf z\mathbf z^H]=\mathbf I_{R_{\rm S}}.
$$

相应的总感知协方差为

$$
\mathbf S=\mathbb E[\mathbf s\mathbf s^H]
=\mathbf V\mathbf V^H
=\sum_{r=1}^{R_{\rm S}}\mathbf v_r\mathbf v_r^H.
$$

因此

$$
\operatorname{rank}(\mathbf S)\le R_{\rm S}.
$$

当 $R_{\rm S}=N_{\rm T}$ 时，任意半正定感知协方差均可表示为 $\mathbf V\mathbf V^H$，理论上不损失表达能力；当 $R_{\rm S}<N_{\rm T}$ 时，该表示是低秩复杂度—性能折中。多目标基准设置可取

$$
R_{\rm S}=\min\{|\mathcal L|,N_{\rm T}\}.
$$

### 假设 A7：采用统一的等效残余 SI 模型

本文不再区分不同处理阶段的残余 SI，而将被动隔离、模拟消除和数字消除后的综合残余效应统一表示为等效残余自干扰信道 $\mathbf H_{\rm RSI}$。该信道进入基站接收信号以及上行通信与目标感知的 SINR 分母，用于反映全双工收发同时工作造成的性能损失。

本模型不设置逐 RF 链路或聚合残余 SI 功率阈值，不引入任何残余 SI 约束、拉格朗日乘子、候选解验收条件或事后检测指标。换言之，残余 SI 只通过加权和速率目标函数影响最优波束、功率和 ToMA 位置。

同时假设综合自干扰消除能力足以保证接收硬件处于正常工作区间；LNA、混频器和 ADC 饱和等硬件安全问题不在本文优化模型中显式描述。

### 假设 A8：简化的 ToMA 几何约束

优化问题仅包含缆绳定长和拖曳子无人机防碰撞约束。部署层假设保证任意发射阵元与接收阵元之间均具有严格正距离，以避免直接 SI 的 $1/d$ 模型奇异。

---

## 2. 单套 ToMA 几何模型

### 2.1 子无人机集合

定义下行用户、上行用户和感知目标集合分别为

$$
\mathcal K=\{1,\ldots,K\},
\qquad
\mathcal J=\{1,\ldots,J\},
\qquad
\mathcal L=\{1,\ldots,L\}.
$$

相应的下行用户、上行用户和感知目标三维位置分别记为

$$
\mathbf q_k^{\rm D},\qquad
\mathbf q_j^{\rm U},\qquad
\mathbf q_l^{\rm S}\in\mathbb R^3,
$$

其中 $k\in\mathcal K$、$j\in\mathcal J$、$l\in\mathcal L$。定义子无人机集合

$$
\mathcal M=\{1,\ldots,M\},
$$

$$
\mathcal M_{\rm T}
=\left\{1,\ldots,\frac M2\right\},
\qquad
\mathcal M_{\rm R}
=\left\{\frac M2+1,\ldots,M\right\}.
$$

满足

$$
\mathcal M_{\rm T}\cap\mathcal M_{\rm R}=\varnothing,
\qquad
\mathcal M_{\rm T}\cup\mathcal M_{\rm R}=\mathcal M.
$$

每根缆绳均匀布置 $N_{\rm c}$ 个阵元，因此

$$
N_{\rm T}=N_{\rm R}=\frac M2N_{\rm c}.
$$

### 2.2 子无人机端点与阵元位置

令

$$
\mathbf c_m\in\mathbb R^3
$$

表示第 $m$ 架拖曳子无人机相对于公共锚点的端点位置。第 $m$ 根缆绳上第 $n$ 个阵元的位置为

$$
\mathbf p_{m,n}=\frac{n}{N_{\rm c}}\mathbf c_m,
\qquad
n=1,\ldots,N_{\rm c}.
$$

为使缆绳编号与全局阵元编号一一对应，对发射缆绳 $m\in\mathcal M_{\rm T}$ 定义

$$
i=(m-1)N_{\rm c}+n,
\qquad
\mathbf p_i^{\rm T}:=\mathbf p_{m,n},
$$

其中 $i=1,\ldots,N_{\rm T}$。对接收缆绳 $m\in\mathcal M_{\rm R}$ 定义

$$
s=\left(m-\frac M2-1\right)N_{\rm c}+n,
\qquad
\mathbf p_s^{\rm R}:=\mathbf p_{m,n},
$$

其中 $s=1,\ldots,N_{\rm R}$。

分别定义发射和接收端点向量

$$
\widetilde{\mathbf c}_{\rm T}
=\left[\mathbf c_m^T\right]_{m\in\mathcal M_{\rm T}}^T,
\qquad
\widetilde{\mathbf c}_{\rm R}
=\left[\mathbf c_m^T\right]_{m\in\mathcal M_{\rm R}}^T.
$$

总位置向量为

$$
\widetilde{\mathbf c}
=
\left[
\widetilde{\mathbf c}_{\rm T}^T,
\widetilde{\mathbf c}_{\rm R}^T
\right]^T.
$$

### 2.3 位置与信道的依赖关系

发射端点位置 $\widetilde{\mathbf c}_{\rm T}$ 决定

$$
\{\mathbf h_k\}_{k\in\mathcal K},
\qquad
\{\mathbf G_l\}_{l\in\mathcal L},
\qquad
\mathbf H_{\rm SI}^{0}.
$$

接收端点位置 $\widetilde{\mathbf c}_{\rm R}$ 决定

$$
\{\mathbf f_j\}_{j\in\mathcal J},
\qquad
\{\mathbf G_l\}_{l\in\mathcal L},
\qquad
\mathbf H_{\rm SI}^{0}.
$$

因此，收发位置通过目标回波和直接 SI 信道相互耦合。

### 2.4 几何可行域

每根缆绳完全展开：

$$
\|\mathbf c_m\|_2=L_{\rm c},
\qquad
\forall m\in\mathcal M.
\tag{C1}
$$

任意两个拖曳子无人机之间满足

$$
\|\mathbf c_m-\mathbf c_{m'}\|_2
\ge D_{\rm UAV},
\qquad
\forall m\ne m'.
\tag{C2}
$$

---

## 3. 近场适用范围

发射与接收阵列的实际有效孔径分别为

$$
D_{\rm T}(\widetilde{\mathbf c}_{\rm T})
=\max_{i,i'}
\|\mathbf p_i^{\rm T}-\mathbf p_{i'}^{\rm T}\|_2,
$$

$$
D_{\rm R}(\widetilde{\mathbf c}_{\rm R})
=\max_{s,s'}
\|\mathbf p_s^{\rm R}-\mathbf p_{s'}^{\rm R}\|_2.
$$

对孔径 $D_x$，$x\in\{\rm T,R\}$，辐射近场近似范围为

$$
r_{{\rm F},x}
=0.62\sqrt{\frac{D_x^3}{\lambda}},
\qquad
r_{{\rm Ray},x}
=\frac{2D_x^2}{\lambda}.
$$

系统配置预先保证用户和目标位于辐射近场。近场条件不作为联合优化约束，而是在算法收敛后根据最终 APV 重新验证。

---

## 4. 纯 LoS 球面波信道

令

$$
k_0=\frac{2\pi}{\lambda}.
$$

### 4.1 收发 USW 阵列响应

对任意空间位置 $\mathbf q$，定义

$$
\mathbf a_{\rm T}(\widetilde{\mathbf c}_{\rm T},\mathbf q)
=
\left[
e^{jk_0\|\mathbf p_i^{\rm T}-\mathbf q\|_2}
\right]_{i=1}^{N_{\rm T}},
$$

$$
\mathbf a_{\rm R}(\widetilde{\mathbf c}_{\rm R},\mathbf q)
=
\left[
e^{jk_0\|\mathbf p_s^{\rm R}-\mathbf q\|_2}
\right]_{s=1}^{N_{\rm R}}.
$$

### 4.2 下行与上行通信信道

第 $k$ 个下行用户的信道为

$$
\mathbf h_k
=
\alpha_k^{\rm D}
\mathbf a_{\rm T}
(\widetilde{\mathbf c}_{\rm T},\mathbf q_k^{\rm D}).
$$

第 $j$ 个上行用户的信道为

$$
\mathbf f_j
=
\alpha_j^{\rm U}
\mathbf a_{\rm R}
(\widetilde{\mathbf c}_{\rm R},\mathbf q_j^{\rm U}).
$$

其中 $\alpha_k^{\rm D}$ 与 $\alpha_j^{\rm U}$ 为对应链路的公共复路径增益。

### 4.3 目标双程回波信道

令 $\xi_l^{\rm S}$ 表示第 $l$ 个目标的复反射系数。进一步令
$\beta_{{\rm T},l}^{\rm S}\ge0$ 和 $\beta_{{\rm R},l}^{\rm S}\ge0$
分别表示发射端至目标以及目标至接收端两段链路的功率增益，因此公共双程复幅度系数中使用其平方根：

$$
\alpha_l^{\rm S}
=
\xi_l^{\rm S}
\sqrt{\beta_{{\rm T},l}^{\rm S}
\beta_{{\rm R},l}^{\rm S}}.
$$

第 $l$ 个目标的双程回波矩阵写为

$$
\boxed{
\mathbf G_l
=
\alpha_l^{\rm S}
\mathbf a_{\rm R}^{*}
(\widetilde{\mathbf c}_{\rm R},\mathbf q_l^{\rm S})
\mathbf a_{\rm T}^{H}
(\widetilde{\mathbf c}_{\rm T},\mathbf q_l^{\rm S})
}.
$$

其第 $(s,i)$ 个元素包含传播相位

$$
e^{-jk_0\left(
\|\mathbf p_s^{\rm R}-\mathbf q_l^{\rm S}\|_2+
\|\mathbf p_i^{\rm T}-\mathbf q_l^{\rm S}\|_2
\right)}.
$$

### 4.4 共锚点直接 SI 信道

第 $i$ 个发射阵元与第 $s$ 个接收阵元的距离为

$$
d_{s,i}^{\rm SI}
=
\|\mathbf p_s^{\rm R}-\mathbf p_i^{\rm T}\|_2.
$$

未消除直接 SI 信道的第 $(s,i)$ 个元素为

$$
\boxed{
[\mathbf H_{\rm SI}^{0}]_{s,i}
=
\frac{\lambda}{4\pi d_{s,i}^{\rm SI}}
e^{-jk_0d_{s,i}^{\rm SI}}
}.
$$

部署层保证

$$
d_{s,i}^{\rm SI}\ge d_{\rm SI}^{\min}>0.
$$

---

## 5. 等效残余 SI 模型

为避免不必要的接收机处理分层，本文仅保留“未消除的物理直接 SI 信道”和“消除后的等效残余 SI 信道”两个概念。令

$$
0\le \rho_{\rm SI}\le1
$$

表示被动隔离、模拟消除和数字消除后的综合残余功率比例，则等效残余 SI 信道定义为

$$
\boxed{
\mathbf H_{\rm RSI}
=
\sqrt{\rho_{\rm SI}}\mathbf H_{\rm SI}^{0}
}.
$$

其中，$\mathbf H_{\rm SI}^{0}$ 由收发 ToMA 阵元之间的几何距离决定，$\rho_{\rm SI}$ 则概括系统整体自干扰抑制能力。$\mathbf H_{\rm RSI}$ 进入基站接收信号以及上行、感知 SINR 的干扰项。本文不再定义前端 SI 信道或基带 SI 信道，不构造任何残余 SI 功率约束，也不利用 SI 功率筛选资源或位置候选解。

---

## 6. 全双工 ISAC 发射信号

### 6.1 通信与感知波束矩阵

令

$$
\mathbf W
=
[\mathbf w_1,\ldots,\mathbf w_K]
\in\mathbb C^{N_{\rm T}\times K}
$$

为下行通信波束矩阵，

$$
\mathbf V
=
[\mathbf v_1,\ldots,\mathbf v_{R_{\rm S}}]
\in\mathbb C^{N_{\rm T}\times R_{\rm S}}
$$

为感知波束矩阵。

定义合并发射波束矩阵

$$
\boxed{
\mathbf Q
=
[\mathbf W,\mathbf V]
\in\mathbb C^{N_{\rm T}\times D}
},
\qquad
D=K+R_{\rm S}.
$$

### 6.2 发射信号

令

$$
\mathbf d
=
[d_1^{\rm D},\ldots,d_K^{\rm D}]^T,
\qquad
\mathbb E[\mathbf d\mathbf d^H]=\mathbf I_K,
$$

且 $\mathbf d$ 与 $\mathbf z$ 相互独立。基站发射信号为

$$
\boxed{
\mathbf x
=
\mathbf W\mathbf d+\mathbf V\mathbf z
=
\sum_{k\in\mathcal K}\mathbf w_kd_k^{\rm D}
+\sum_{r=1}^{R_{\rm S}}\mathbf v_rz_r
}.
$$

总发射协方差为

$$
\boxed{
\mathbf R_x
=
\mathbb E[\mathbf x\mathbf x^H]
=
\mathbf W\mathbf W^H+\mathbf V\mathbf V^H
=
\mathbf Q\mathbf Q^H
}.
$$

总发射功率为

$$
\operatorname{Tr}(\mathbf R_x)
=
\|\mathbf Q\|_F^2.
$$

---

## 7. 接收信号模型

### 7.1 下行用户接收信号

第 $k$ 个下行用户接收

$$
y_k^{\rm D}
=
\mathbf h_k^H\mathbf w_kd_k^{\rm D}
+
\sum_{i\in\mathcal K\setminus\{k\}}
\mathbf h_k^H\mathbf w_id_i^{\rm D}
+
\mathbf h_k^H\mathbf V\mathbf z
+n_k.
$$

其中下行用户噪声满足

$$
n_k\sim\mathcal{CN}(0,\sigma_k^2).
$$

### 7.2 基站接收信号

令

$$
p_j=q_j^2,
\qquad
q_j\ge0,
$$

且 $\mathbb E[|d_j^{\rm U}|^2]=1$。经过综合自干扰消除后的基站接收信号为

$$
\boxed{
\mathbf y_{\rm BS}
=
\sum_{j\in\mathcal J}q_j\mathbf f_jd_j^{\rm U}
+
\sum_{l\in\mathcal L}\mathbf G_l\mathbf x
+
\mathbf H_{\rm RSI}\mathbf x
+
\mathbf n_{\rm BS}
}.
$$

其中基站接收噪声满足

$$
\mathbf n_{\rm BS}
\sim
\mathcal{CN}
(\mathbf0,\sigma_{\rm BS}^2\mathbf I_{N_{\rm R}}).
$$

虽然目标回波与残余 SI 在接收信号层面叠加，但本文在 SINR 的二阶统计建模中将不同目标回波和残余 SI 视为互不相关的干扰分量。因此，各分量的协方差在功率域分别累加，不保留不同回波之间以及回波与残余 SI 之间的交叉协方差项。该处理避免把依赖瞬时相位的 echo--SI 相干抵消解释为稳定的干扰抑制增益，同时仍保留 $\mathbf H_{\rm RSI}=\mathbf H_{\rm RSI}(C)$ 对收发阵列几何和发射协方差的依赖。

---

## 8. 通信与感知性能

### 8.1 下行 SINR 与速率

利用

$$
\|\mathbf h_k^H\mathbf V\|_2^2
=
\sum_{r=1}^{R_{\rm S}}
|\mathbf h_k^H\mathbf v_r|^2,
$$

第 $k$ 个下行用户的 SINR 为

$$
\boxed{
\gamma_k^{\rm D}
=
\frac{|\mathbf h_k^H\mathbf w_k|^2}
{
\displaystyle
\sum_{i\ne k}|\mathbf h_k^H\mathbf w_i|^2
+
\|\mathbf h_k^H\mathbf V\|_2^2
+\sigma_k^2
}
}.
$$

速率为

$$
R_k^{\rm D}
=
\log_2(1+\gamma_k^{\rm D}).
$$

### 8.2 上行 SINR 与速率

令 $\|\mathbf b_j\|_2=1$，并定义上行回波与残余 SI 的干扰协方差

$$
\boxed{
\mathbf C_{\rm U}
=
\sum_{l\in\mathcal L}
\mathbf G_l\mathbf R_x\mathbf G_l^H
+
\mathbf H_{\rm RSI}\mathbf R_x\mathbf H_{\rm RSI}^H
}.
$$

第 $j$ 个上行用户的 SINR 为

$$
\boxed{
\gamma_j^{\rm U}
=
\frac{
q_j^2|\mathbf b_j^H\mathbf f_j|^2
}{
\displaystyle
\sum_{i\ne j}
q_i^2|\mathbf b_j^H\mathbf f_i|^2
+
\mathbf b_j^H\mathbf C_{\rm U}\mathbf b_j
+\sigma_{\rm BS}^2
}
}.
$$

速率为

$$
R_j^{\rm U}
=
\log_2(1+\gamma_j^{\rm U}).
$$

### 8.3 感知 SINR 与感知速率

令 $\|\mathbf u_l\|_2=1$，并定义目标 $l$ 对应的其他回波与残余 SI 干扰协方差

$$
\boxed{
\mathbf C_{{\rm S},l}
=
\sum_{i\in\mathcal L\setminus\{l\}}
\mathbf G_i\mathbf R_x\mathbf G_i^H
+
\mathbf H_{\rm RSI}\mathbf R_x\mathbf H_{\rm RSI}^H
}.
$$

目标 $l$ 的期望回波功率为

$$
\|\mathbf u_l^H\mathbf G_l\mathbf Q\|_2^2.
$$

相应感知 SINR 为

$$
\boxed{
\gamma_l^{\rm S}
=
\frac{
\|\mathbf u_l^H\mathbf G_l\mathbf Q\|_2^2
}{
\displaystyle
\sum_{j\in\mathcal J}
q_j^2|\mathbf u_l^H\mathbf f_j|^2
+
\mathbf u_l^H\mathbf C_{{\rm S},l}\mathbf u_l
+\sigma_{\rm BS}^2
}
}.
$$

感知速率定义为

$$
R_l^{\rm S}
=
\log_2(1+\gamma_l^{\rm S}).
$$

---

## 9. 等效残余 SI 的处理原则

本模型对残余 SI 采用“只进入性能表达式、不进入可行域”的处理方式：

1. $\mathbf H_{\rm RSI}\mathbf x$ 保留在基站接收信号中；
2. $\mathbf H_{\rm RSI}$ 保留在上行与感知 SINR 的干扰项中；
3. 不设置逐 RF 链路 SI 阈值、总残余 SI 阈值或波束合并后的 SI 阈值；
4. 不执行资源候选解 SI 检查、位置候选解 SI 检查或最终 SI 合格判定。

因此，残余 SI 对优化结果的影响完全由加权和速率目标函数体现。例如，当某一发射波束或 ToMA 位置使残余 SI 增大并显著降低上行或感知速率时，算法可能通过最大化总性能而主动避开该方案；但这种抑制是目标函数驱动的性能权衡，而不是由硬阈值强制产生。

---

## 10. 联合性能优化问题

定义优化变量

$$
\boldsymbol\Xi
=
\left\{
\widetilde{\mathbf c}_{\rm T},
\widetilde{\mathbf c}_{\rm R},
\mathbf Q,
\{\mathbf u_l\},
\{\mathbf b_j\},
\{q_j\}
\right\}.
$$

加权和速率为

$$
F(\boldsymbol\Xi)
=
\sum_{l\in\mathcal L}\omega_l^{\rm S}R_l^{\rm S}
+
\sum_{j\in\mathcal J}\omega_j^{\rm U}R_j^{\rm U}
+
\sum_{k\in\mathcal K}\omega_k^{\rm D}R_k^{\rm D},
$$

其中

$$
\omega_l^{\rm S},\omega_j^{\rm U},\omega_k^{\rm D}\ge0,
$$

$$
\sum_l\omega_l^{\rm S}
+
\sum_j\omega_j^{\rm U}
+
\sum_k\omega_k^{\rm D}=1.
$$

删除全部残余 SI 约束和检测条件后，联合优化问题写为

$$
\begin{aligned}
\text{(P1)}\quad
\max_{\boldsymbol\Xi}\quad
&F(\boldsymbol\Xi)\\
\text{s.t.}\quad
&\|\mathbf u_l\|_2^2=1,
\quad \forall l,\\
&\|\mathbf b_j\|_2^2=1,
\quad \forall j,\\
&\|\mathbf Q\|_F^2\le P_{\rm D}^{\max},\\
&0\le q_j\le\sqrt{P_{{\rm U},j}^{\max}},
\quad \forall j,\\
&\|\mathbf c_m\|_2=L_{\rm c},
\quad \forall m,\\
&\|\mathbf c_m-\mathbf c_{m'}\|_2
\ge D_{\rm UAV},
\quad \forall m\ne m'.
\end{aligned}
$$

(P1) 的可行域仅由接收波束归一化、上下行功率预算和 ToMA 几何条件决定。残余 SI 不单独限制可行解，而是通过 $R_j^{\rm U}$ 和 $R_l^{\rm S}$ 进入目标函数。

---

## 11. 问题性质与两层算法框架

(P1) 的主要困难包括：

1. 多个通信与感知速率含相互耦合的分式 SINR；
2. 发射矩阵 $\mathbf Q$、上行幅度 $\{q_j\}$ 和接收波束相互耦合；
3. 收发 ToMA 端点位于球面流形，并受非凸防碰撞约束；
4. 球面波通信信道、目标回波信道和直接 SI 信道均随 APV 非线性变化。

采用两层交替优化：

$$
\boxed{
\underbrace{
\{\mathbf u_l,\mathbf b_j\}
\leftrightarrow
\{\mathbf Q,q_j,\text{FP辅助变量}\}
}_{\text{内层：FP＋拉格朗日}}
\longrightarrow
\underbrace{
\widetilde{\mathbf c}_{\rm T}
\longrightarrow
\widetilde{\mathbf c}_{\rm R}
}_{\text{外层：球面黎曼优化}}
}.
$$

内层不使用 SDP，也不引入通信协方差变量
$\mathbf W_k=\mathbf w_k\mathbf w_k^H$，因此不存在 SDR 和秩一恢复步骤。接收波束、QT 总功率和发射矩阵二次项均按第 7.2 节的互不相关协方差假设展开；上行功率更新的结构保持不变。模型仍不包含 SI 约束验收和相关回溯。

外层采用 FP-aware 黎曼位置更新：位置梯度仍在当前资源固定条件下计算，但每个可行候选位置均以当前资源为 warm start，在候选信道上执行最多 2 次内层 FP 更新后再进行 Armijo 验收；一旦验收通过，候选位置及其候选资源同步进入下一端点更新。该步骤仅改变候选点评价方式，不改变 (P1)、内层 FP 更新或 RCG 主体。

---

## 12. 内层步骤一：接收波束更新

固定 $\mathbf Q$、$\{q_j\}$ 和全部位置变量。

### 12.1 感知接收波束

定义

$$
\mathbf C_l^{\rm S}
=
\mathbf G_l\mathbf Q\mathbf Q^H\mathbf G_l^H,
$$

$$
\mathbf D_l^{\rm S}
=
\sum_{j\in\mathcal J}
q_j^2\mathbf f_j\mathbf f_j^H
+
\mathbf C_{{\rm S},l}
+
\sigma_{\rm BS}^2\mathbf I_{N_{\rm R}}.
$$

最大 SINR 感知接收波束为

$$
\boxed{
\mathbf u_l^\star
=
\frac{\widehat{\mathbf u}_l}
{\|\widehat{\mathbf u}_l\|_2},
\qquad
\widehat{\mathbf u}_l
=
\operatorname{eigvec}_{\max}
(\mathbf C_l^{\rm S},\mathbf D_l^{\rm S})
}.
$$

### 12.2 上行接收波束

定义

$$
\mathbf D_j^{\rm U}
=
\sum_{i\ne j}q_i^2\mathbf f_i\mathbf f_i^H
+
\mathbf C_{\rm U}
+
\sigma_{\rm BS}^2\mathbf I_{N_{\rm R}}.
$$

最大 SINR/MMSE 方向为

$$
\boxed{
\mathbf b_j^\star
=
\frac{
(\mathbf D_j^{\rm U})^{-1}\mathbf f_j
}{
\|(\mathbf D_j^{\rm U})^{-1}\mathbf f_j\|_2
}
}.
$$

数值实现采用线性方程求解，不显式计算矩阵逆。

---


> **符号说明：** 信道公共复路径增益继续记为 $\alpha_k^{\rm D}$、$\alpha_j^{\rm U}$ 和 $\alpha_l^{\rm S}$；为避免符号冲突，LDT 辅助变量统一记为 $\eta_k^{\rm D}$、$\eta_j^{\rm U}$ 和 $\eta_l^{\rm S}$。

## 13. 内层步骤二：LDT 分式规划变换

本节采用分式规划（Fractional Programming，FP）处理加权和速率中的分式 SINR；具体依次使用拉格朗日对偶变换（Lagrangian Dual Transform，LDT）和后续二次变换（Quadratic Transform，QT）。

### 13.1 拉格朗日对偶变换恒等式

对任意 $A\ge0$、$B>0$，

$$
\log_2\left(1+\frac AB\right)
=
\frac1{\ln2}
\max_{\eta\ge0}
\left[
\ln(1+\eta)-\eta
+
(1+\eta)\frac{A}{A+B}
\right].
$$

最优辅助变量为

$$
\boxed{\eta^\star=\frac AB=\gamma}.
$$

这里的三类速率原本均以 $\log_2(1+\gamma)$ 表示。利用换底公式

$$
\log_2(1+\gamma)
=
\frac{\ln(1+\gamma)}{\ln2},
$$

定义换底后的三类权重

$$
\bar\omega_k^{\rm D}
:=
\frac{\omega_k^{\rm D}}{\ln2},
\qquad
\bar\omega_j^{\rm U}
:=
\frac{\omega_j^{\rm U}}{\ln2},
\qquad
\bar\omega_l^{\rm S}
:=
\frac{\omega_l^{\rm S}}{\ln2}.
$$

因此，原加权速率项可分别改写为

$$
\omega_k^{\rm D}R_k^{\rm D}
=
\bar\omega_k^{\rm D}\ln(1+\gamma_k^{\rm D}),
$$

$$
\omega_j^{\rm U}R_j^{\rm U}
=
\bar\omega_j^{\rm U}\ln(1+\gamma_j^{\rm U}),
\qquad
\omega_l^{\rm S}R_l^{\rm S}
=
\bar\omega_l^{\rm S}\ln(1+\gamma_l^{\rm S}).
$$

后续 LDT 和 QT 公式中的 $\bar\omega_k^{\rm D}$、$\bar\omega_j^{\rm U}$ 与
$\bar\omega_l^{\rm S}$ 均指上述换底权重，并非新增的独立优化变量。

### 13.2 三类总接收功率

下行用户 $k$ 的总接收功率为

$$
T_k^{\rm D}
=
\|\mathbf h_k^H\mathbf Q\|_2^2+\sigma_k^2.
$$

上行用户 $j$ 经 $\mathbf b_j$ 合并后的总接收功率为

$$
T_j^{\rm U}
=
\sum_{i\in\mathcal J}
q_i^2|\mathbf b_j^H\mathbf f_i|^2
+
\mathbf b_j^H\mathbf C_{\rm U}\mathbf b_j
+
\sigma_{\rm BS}^2.
$$

目标 $l$ 经 $\mathbf u_l$ 合并后的总接收功率为

$$
\begin{aligned}
T_l^{\rm S}
={}&
\|\mathbf u_l^H\mathbf G_l\mathbf Q\|_2^2\\
&+
\sum_{j\in\mathcal J}
q_j^2|\mathbf u_l^H\mathbf f_j|^2\\
&+
\mathbf u_l^H\mathbf C_{{\rm S},l}\mathbf u_l
+
\sigma_{\rm BS}^2.
\end{aligned}
$$

---

## 14. 内层步骤三：二次变换辅助变量

### 14.1 下行标量二次变换

定义标量辅助变量 $\phi_k^{\rm D}\in\mathbb C$。有

$$
\frac{|\mathbf h_k^H\mathbf w_k|^2}{T_k^{\rm D}}
=
\max_{\phi_k^{\rm D}}
\left[
2\operatorname{Re}
\left\{
(\phi_k^{\rm D})^*
\mathbf h_k^H\mathbf w_k
\right\}
-
|\phi_k^{\rm D}|^2T_k^{\rm D}
\right].
$$

最优更新为

$$
\boxed{
\phi_k^{{\rm D}\star}
=
\frac{\mathbf h_k^H\mathbf w_k}
{T_k^{\rm D}}
}.
$$

### 14.2 上行标量二次变换

定义 $\phi_j^{\rm U}\in\mathbb C$。有

$$
\frac{
q_j^2|\mathbf b_j^H\mathbf f_j|^2
}{
T_j^{\rm U}
}
=
\max_{\phi_j^{\rm U}}
\left[
2\operatorname{Re}
\left\{
(\phi_j^{\rm U})^*
q_j\mathbf b_j^H\mathbf f_j
\right\}
-
|\phi_j^{\rm U}|^2T_j^{\rm U}
\right].
$$

最优更新为

$$
\boxed{
\phi_j^{{\rm U}\star}
=
\frac{
q_j\mathbf b_j^H\mathbf f_j
}{
T_j^{\rm U}
}
}.
$$

### 14.3 感知向量二次变换

由于目标 $l$ 的期望回波来自全部通信和感知发射流，定义

$$
\mathbf g_l
=
\mathbf G_l^H\mathbf u_l
\in\mathbb C^{N_{\rm T}},
$$

以及向量辅助变量

$$
\boldsymbol\phi_l^{\rm S}
\in\mathbb C^{D}.
$$

有

$$
\frac{
\|\mathbf u_l^H\mathbf G_l\mathbf Q\|_2^2
}{
T_l^{\rm S}
}
=
\max_{\boldsymbol\phi_l^{\rm S}}
\left[
2\operatorname{Re}
\left\{
(\boldsymbol\phi_l^{\rm S})^H
\mathbf Q^H\mathbf g_l
\right\}
-
\|\boldsymbol\phi_l^{\rm S}\|_2^2T_l^{\rm S}
\right].
$$

最优更新为

$$
\boxed{
\boldsymbol\phi_l^{{\rm S}\star}
=
\frac{
\mathbf Q^H\mathbf G_l^H\mathbf u_l
}{
T_l^{\rm S}
}
}.
$$

---

## 15. 内层步骤四：发射矩阵的拉格朗日闭式更新

固定接收波束、$\{\eta\}$、$\{\phi^{\rm D}\}$、$\{\phi^{\rm U}\}$、$\{\boldsymbol\phi^{\rm S}\}$ 和 $\{q_j\}$。

### 15.1 二次矩阵项

定义

$$
\mathbf C_k^{\rm D}
=
\mathbf h_k\mathbf h_k^H,
$$

$$
\mathbf C_j^{\rm U}
=
\sum_{i\in\mathcal L}
\mathbf G_i^H\mathbf b_j\mathbf b_j^H\mathbf G_i
+
\mathbf H_{\rm RSI}^H
\mathbf b_j\mathbf b_j^H
\mathbf H_{\rm RSI},
$$

$$
\mathbf C_{l,{\rm des}}^{\rm S}
=
\mathbf G_l^H
\mathbf u_l\mathbf u_l^H
\mathbf G_l,
$$

$$
\mathbf C_{l,{\rm int}}^{\rm S}
=
\sum_{i\in\mathcal L\setminus\{l\}}
\mathbf G_i^H\mathbf u_l\mathbf u_l^H\mathbf G_i
+
\mathbf H_{\rm RSI}^H
\mathbf u_l\mathbf u_l^H
\mathbf H_{\rm RSI}.
$$

收集所有关于 $\mathbf Q$ 的负二次项：

$$
\begin{aligned}
\mathbf A_Q
={}&
\sum_{k\in\mathcal K}
\bar\omega_k^{\rm D}
(1+\eta_k^{\rm D})
|\phi_k^{\rm D}|^2
\mathbf C_k^{\rm D}\\
&+
\sum_{j\in\mathcal J}
\bar\omega_j^{\rm U}
(1+\eta_j^{\rm U})
|\phi_j^{\rm U}|^2
\mathbf C_j^{\rm U}\\
&+
\sum_{l\in\mathcal L}
\bar\omega_l^{\rm S}
(1+\eta_l^{\rm S})
\|\boldsymbol\phi_l^{\rm S}\|_2^2
\left(
\mathbf C_{l,{\rm des}}^{\rm S}
+
\mathbf C_{l,{\rm int}}^{\rm S}
\right).
\end{aligned}
$$

显然

$$
\mathbf A_Q\succeq\mathbf0.
$$

### 15.2 线性矩阵项

令 $\mathbf e_k^{(D)}\in\mathbb R^D$ 为第 $k$ 个标准基向量，用于选择 $\mathbf Q$ 的第 $k$ 个通信波束列。定义

$$
\begin{aligned}
\mathbf B_Q
={}&
\sum_{k\in\mathcal K}
\bar\omega_k^{\rm D}
(1+\eta_k^{\rm D})
\phi_k^{\rm D}
\mathbf h_k
(\mathbf e_k^{(D)})^T\\
&+
\sum_{l\in\mathcal L}
\bar\omega_l^{\rm S}
(1+\eta_l^{\rm S})
\mathbf G_l^H\mathbf u_l
(\boldsymbol\phi_l^{\rm S})^H.
\end{aligned}
$$

去除与 $\mathbf Q$ 无关的常数后，发射矩阵子问题为

$$
\begin{aligned}
\max_{\mathbf Q}\quad
&
2\operatorname{Re}
\{\operatorname{Tr}(\mathbf B_Q^H\mathbf Q)\}
-
\operatorname{Tr}(\mathbf Q^H\mathbf A_Q\mathbf Q)\\
\text{s.t.}\quad
&\|\mathbf Q\|_F^2\le P_{\rm D}^{\max}.
\end{aligned}
\tag{P-Q}
$$

### 15.3 总功率拉格朗日乘子

为总发射功率约束引入一个乘子

$$
\lambda_{\rm P}\ge0.
$$

拉格朗日函数为

$$
\begin{aligned}
\mathcal L_Q
={}&
2\operatorname{Re}
\{\operatorname{Tr}(\mathbf B_Q^H\mathbf Q)\}
-
\operatorname{Tr}(\mathbf Q^H\mathbf A_Q\mathbf Q)\\
&-
\lambda_{\rm P}
\left(
\|\mathbf Q\|_F^2-P_{\rm D}^{\max}
\right).
\end{aligned}
$$

一阶最优条件给出

$$
\boxed{
\widehat{\mathbf Q}
=
(\mathbf A_Q+\lambda_{\rm P}\mathbf I_{N_{\rm T}})^{-1}
\mathbf B_Q
}.
$$

若

$$
\|\mathbf A_Q^\dagger\mathbf B_Q\|_F^2
\le P_{\rm D}^{\max},
$$

可取 $\lambda_{\rm P}=0$。否则通过一维二分搜索寻找唯一的 $\lambda_{\rm P}>0$，使

$$
\|(\mathbf A_Q+\lambda_{\rm P}\mathbf I_{N_{\rm T}})^{-1}
\mathbf B_Q\|_F^2
=
P_{\rm D}^{\max}.
$$

得到 $\widehat{\mathbf Q}$ 后，按列分解为

$$
\widehat{\mathbf Q}
=
[
\widehat{\mathbf w}_1,\ldots,\widehat{\mathbf w}_K,
\widehat{\mathbf v}_1,\ldots,\widehat{\mathbf v}_{R_{\rm S}}
].
$$

每个通信波束和感知波束直接由向量表示，不需要秩一恢复。

---

## 16. 内层步骤五：上行功率幅度闭式更新

固定 $\mathbf Q$ 和全部辅助变量。关于 $q_j$ 的目标可写成

$$
2\mu_jq_j-\nu_jq_j^2+\text{常数},
$$

其中

$$
\mu_j
=
\bar\omega_j^{\rm U}
(1+\eta_j^{\rm U})
\operatorname{Re}
\left\{
(\phi_j^{\rm U})^*
\mathbf b_j^H\mathbf f_j
\right\},
$$

$$
\begin{aligned}
\nu_j
={}&
\sum_{i\in\mathcal J}
\bar\omega_i^{\rm U}
(1+\eta_i^{\rm U})
|\phi_i^{\rm U}|^2
|\mathbf b_i^H\mathbf f_j|^2\\
&+
\sum_{l\in\mathcal L}
\bar\omega_l^{\rm S}
(1+\eta_l^{\rm S})
\|\boldsymbol\phi_l^{\rm S}\|_2^2
|\mathbf u_l^H\mathbf f_j|^2.
\end{aligned}
$$

定义上行幅度上界

$$
q_{j,\max}:=\sqrt{P_{{\rm U},j}^{\max}}.
$$

当 $\nu_j>0$ 时，根据无约束驻点 $\mu_j/\nu_j$ 是否落在合法区间
$[0,q_{j,\max}]$ 内，最优候选幅度可分类写为

$$
\boxed{
\widehat q_j=
\begin{cases}
0,
& \mu_j\le0,\\[2mm]
\dfrac{\mu_j}{\nu_j},
& 0<\mu_j<\nu_jq_{j,\max},\\[3mm]
q_{j,\max},
& \mu_j\ge \nu_jq_{j,\max}.
\end{cases}
}
$$

代入 $q_{j,\max}=\sqrt{P_{{\rm U},j}^{\max}}$，即

$$
\boxed{
\widehat q_j=
\begin{cases}
0,
& \mu_j\le0,\\[2mm]
\dfrac{\mu_j}{\nu_j},
& 0<\mu_j<\nu_j\sqrt{P_{{\rm U},j}^{\max}},\\[3mm]
\sqrt{P_{{\rm U},j}^{\max}},
& \mu_j\ge \nu_j\sqrt{P_{{\rm U},j}^{\max}}.
\end{cases}
}
$$

若 $\nu_j=0$，目标关于 $q_j$ 退化为线性函数 $2\mu_jq_j$：当 $\mu_j<0$ 时取
$\widehat q_j=0$，当 $\mu_j>0$ 时取 $\widehat q_j=q_{j,\max}$；当 $\mu_j=0$ 时区间
$[0,q_{j,\max}]$ 内任意值均为最优，可约定取 $\widehat q_j=0$。实际数值中也可为
$\nu_j$ 加入极小正数以避免除零。最终上行功率为

$$
p_j=\widehat q_j^2.
$$

---

## 17. 内层资源变量的直接更新

固定 APV 以及当前接收波束和 FP 辅助变量后，第 15 节得到满足总发射功率约束的

$$
\widehat{\mathbf Q}
=
(\mathbf A_Q+\lambda_{\rm P}\mathbf I_{N_{\rm T}})^{-1}\mathbf B_Q,
$$

第 16 节得到满足上行幅度区间约束的 $\{\widehat q_j\}$。由于模型中不存在残余 SI 约束或 SI 候选验收条件，两个资源块可直接更新为

令 $n$ 表示内层迭代编号，则

$$
\boxed{
\mathbf Q^{(n+1)}=\widehat{\mathbf Q}
},
$$

$$
\boxed{
q_j^{(n+1)}=\widehat q_j,
\qquad \forall j\in\mathcal J
}.
$$

其中，$\lambda_{\rm P}$ 的一维搜索保证

$$
\|\mathbf Q^{(n+1)}\|_F^2\le P_{\rm D}^{\max},
$$

而第 16 节的分段闭式解保证

$$
0\le q_j^{(n+1)}\le\sqrt{P_{{\rm U},j}^{\max}}.
$$

因此，内层不再需要针对 SI 的阻尼、回溯、拒绝或恢复旧解步骤。更新后重新计算接收波束、辅助变量、真实 SINR 和真实加权和速率即可。

---

## 18. 内层 FP－拉格朗日交替流程

在固定 APV 下，内层按以下顺序迭代：

1. 根据当前 $\mathbf Q$ 和 $\{q_j\}$ 更新全部感知接收波束 $\{\mathbf u_l\}$；
2. 更新全部上行接收波束 $\{\mathbf b_j\}$；
3. 计算真实 SINR，并更新
$$
   \eta_k^{\rm D}=\gamma_k^{\rm D},
   \quad
   \eta_j^{\rm U}=\gamma_j^{\rm U},
   \quad
   \eta_l^{\rm S}=\gamma_l^{\rm S};
$$
4. 更新二次变换辅助变量
$$
   \{\phi_k^{\rm D}\},
   \quad
   \{\phi_j^{\rm U}\},
   \quad
   \{\boldsymbol\phi_l^{\rm S}\};
$$
5. 构造 $\mathbf A_Q$ 和 $\mathbf B_Q$，利用总功率拉格朗日乘子更新 $\mathbf Q$；
6. 利用分段闭式解更新 $\{q_j\}$；
7. 重新计算真实 SINR 与真实加权和速率 $F_{\rm in}^{(n+1)}$；
8. 当内层相对增益满足
$$
   \frac{
   |F_{\rm in}^{(n+1)}-F_{\rm in}^{(n)}|
   }{
   \max\{1,|F_{\rm in}^{(n)}|\}
   }
   \le\varepsilon_{\rm in}
$$
   时停止。

整个流程中不计算任何 SI 阈值比值，也不利用残余 SI 功率决定候选解是否被接受。

---

## 19. 外层 ToMA 球面黎曼优化

在每一个 ToMA 端点更新开始时，将该时刻的完整 ToMA 位置和当前资源分别记为 $C_t$ 与 $R_t$，其中 $R_t$ 包含此时的

$$
\mathbf Q,\quad
\{q_j\},\quad
\{\mathbf u_l\},\quad
\{\mathbf b_j\}.
$$

在该端点的中央差分、黎曼梯度投影和 RCG 搜索方向构造过程中，$R_t$ 保持固定。若前一个端点的 FP-aware 候选更新已被接受，则同步接受的候选位置与候选资源构成当前端点新的 $C_t$ 与 $R_t$；因此，不同端点更新所对应的当前资源可以不同。该端点的位置梯度为

$$
g_t
=
\operatorname{grad}_{C}F(C_t,R_t),
$$

该梯度是固定该时刻当前资源 $R_t$ 的位置梯度，而不是严格约化目标 $\Phi(C)=F(C,R^\star(C))$ 的梯度。候选位置处的少量 FP 更新仅用于第 20 节的回溯验收。

依次更新

$$
\mathbf c_1\rightarrow\cdots\rightarrow\mathbf c_{M/2}
\rightarrow
\mathbf c_{M/2+1}\rightarrow\cdots\rightarrow\mathbf c_M.
$$

### 19.1 球面切空间

第 $m$ 个端点位于半径 $L_{\rm c}$ 的球面，其切空间为

$$
T_{\mathbf c_m}\mathcal S_m
=
\{
\mathbf v\in\mathbb R^3:
\mathbf c_m^T\mathbf v=0
\}.
$$

将欧氏梯度投影到切空间：

$$
\boxed{
\mathbf g_m
=
\left(
\mathbf I_3-
\frac{\mathbf c_m\mathbf c_m^T}{L_{\rm c}^2}
\right)
\nabla_{\mathbf c_m}F
}.
$$

### 19.2 中央数值差分

对 $a\in\{1,2,3\}$，

$$
[\nabla_{\mathbf c_m}F]_a
\approx
\frac{
F(\mathbf c_m+\delta\mathbf e_a)
-
F(\mathbf c_m-\delta\mathbf e_a)
}{
2\delta
}.
$$

每次函数评估都根据扰动后的端点重新生成阵元位置和相关信道。差分点只用于估计梯度，不作为可接受位置。

上述差分中的资源始终固定为该端点更新开始时的 $R_t$；因此，中央差分、切空间投影、PR+ 系数、向量传输和 RCG 搜索方向均保持原定义不变。

### 19.3 黎曼共轭方向

采用 Polak–Ribiere+ 方向

$$
\boldsymbol\psi_m^{(t)}
=
\mathbf g_m^{(t)}
+
\beta_m^{(t)}
\mathcal T
(\boldsymbol\psi_m^{(t-1)}),
$$

$$
\beta_m^{(t)}
=
\max\left\{
0,
\frac{
\langle
\mathbf g_m^{(t)},
\mathbf g_m^{(t)}
-
\mathcal T(\mathbf g_m^{(t-1)})
\rangle
}{
\|\mathbf g_m^{(t-1)}\|_2^2
}
\right\}.
$$

若

$$
\langle
\mathbf g_m^{(t)},
\boldsymbol\psi_m^{(t)}
\rangle
\le0,
$$

则重启为

$$
\boldsymbol\psi_m^{(t)}=\mathbf g_m^{(t)}.
$$

由于联合迭代过程中资源变量会随内层 FP 以及已接受的 FP-aware 端点更新而变化，跨 Joint AO 轮次保留的前一轮黎曼梯度与搜索方向一般对应于不同的当前资源状态，即通常有 $R_t\ne R_{t-1}$。因此，本文的 PR+ 历史项主要用于保留位置搜索的历史方向信息并加速几何更新，不将其解释为严格固定单一目标函数上的标准 RCG 共轭序列，也不将 $\beta_m^{(t)}$ 解释为严格约化目标 $\Phi(C)$ 的 PR+ 共轭系数。若 PR+ 构造的方向不满足 $\langle\mathbf g_m^{(t)},\boldsymbol\psi_m^{(t)}\rangle>0$，则按上述规则立即重启为 $\boldsymbol\psi_m^{(t)}=\mathbf g_m^{(t)}$，从而保证当前搜索方向仍为上升方向。

### 19.4 球面回缩

给定试探步长 $\tau_m$，候选端点为

$$
\boxed{
\mathbf c_m^{\rm cand}(\tau_m)
=
L_{\rm c}
\frac{
\mathbf c_m+\tau_m\boldsymbol\psi_m
}{
\|\mathbf c_m+\tau_m\boldsymbol\psi_m\|_2
}
}.
$$

该回缩自动满足缆绳定长约束。

---

## 20. 外层候选位置的 FP-aware 可行 Armijo 回溯

每个候选位置需通过几何可行性检查和 FP-aware 目标函数上升条件。梯度仍按第 19 节固定 $R_t$ 计算；仅候选点评价允许资源在新信道上进行少量重新适配。

### 20.1 防碰撞条件

$$
\|\mathbf c_m^{\rm cand}-\mathbf c_{m'}\|_2
\ge D_{\rm UAV},
\qquad
\forall m'\ne m.
$$

### 20.2 Armijo 上升条件

由球面回缩得到完整候选位置 $C_{\rm cand}$（其中仅第 $m$ 个端点替换为 $\mathbf c_m^{\rm cand}$）后，重新生成候选几何与全部相关信道，并以当前资源 $R_t$ 为 warm start，在候选信道上运行最多 2 次内层 FP 更新。记

$$
\boxed{
\widetilde R_{\rm cand}
=
\mathcal T_{\rm FP}^{(2)}
(C_{\rm cand};R_t)
},
$$

其中上标 $(2)$ 表示内层 FP 的迭代上限为 2；若提前满足原内层停止条件，实际更新次数可以少于 2。定义 FP-aware 候选目标

$$
\boxed{
\widehat\Phi_2(C_{\rm cand};R_t)
=
F(C_{\rm cand},\widetilde R_{\rm cand})
}.
$$

相应的 Armijo 上升条件为

$$
\boxed{
\widehat\Phi_2(C_{\rm cand};R_t)
\ge
F(C_t,R_t)
+
\mu_{\rm A}\tau_m
\langle\mathbf g_m,\boldsymbol\psi_m\rangle,
\qquad
0<\mu_{\rm A}<1.
}
$$

若条件成立，则位置与候选资源必须同步接受：

$$
\boxed{
C_{t+1}=C_{\rm cand},
\qquad
R_{t+1}=\widetilde R_{\rm cand}
}.
$$

因此，已接受的新位置不会与旧资源 $R_t$ 重新组合；同步接受的 $R_{t+1}$ 也作为顺序更新下一个端点时的当前资源和 warm start。

若防碰撞条件或 Armijo 条件失败，则

$$
\tau_m\leftarrow\zeta_{\rm out}\tau_m,
\qquad
0<\zeta_{\rm out}<1,
$$

并重新生成候选端点。若达到最大回溯次数仍未通过验收，则同时保留当前位置 $C_t$ 与当前资源 $R_t$。本文不再对候选位置执行残余 SI 功率检查。

发射端点更新后重算

$$
\{\mathbf h_k\},
\quad
\{\mathbf G_l\},
\quad
\mathbf H_{\rm RSI}.
$$

接收端点更新后重算

$$
\{\mathbf f_j\},
\quad
\{\mathbf G_l\},
\quad
\mathbf H_{\rm RSI}.
$$

实际候选点评价会重建候选几何和完整信道状态。由于 $\mathbf H_{\rm RSI}$ 随收发阵元距离变化，它仍需在每个位置候选的目标函数评估中更新；但它仅用于计算真实上行和感知速率，不用于可行性判定。

这里的 $\widehat\Phi_2(C_{\rm cand};R_t)$ 只是在有限次 warm-start FP 更新下对约化目标 $\Phi(C_{\rm cand})=F(C_{\rm cand},R^\star(C_{\rm cand}))$ 的低成本近似，二者不要求严格相等。因此，该方法称为 FP-aware 黎曼位置更新（或 FP-aware 非精确约化目标回溯），而不是严格的约化目标 RCG；算法也未计算 $\nabla\Phi(C)$。

---

## 21. 可行初始化

### 21.1 位置初始化

生成多组满足

$$
\|\mathbf c_m\|_2=L_{\rm c}
$$

和

$$
\|\mathbf c_m-\mathbf c_{m'}\|_2
\ge D_{\rm UAV}
$$

的 APV。

### 21.2 通信与感知波束初始化

下行通信方向采用 MRT：

$$
\overline{\mathbf w}_k
=
\frac{\mathbf h_k}{\|\mathbf h_k\|_2}.
$$

感知波束可采用目标发射阵列响应初始化。例如取

$$
\overline{\mathbf v}_r
=
\frac{
\mathbf a_{\rm T}
(\widetilde{\mathbf c}_{\rm T},\mathbf q_r^{\rm S})
}{
\|
\mathbf a_{\rm T}
(\widetilde{\mathbf c}_{\rm T},\mathbf q_r^{\rm S})
\|_2
},
\qquad
r=1,\ldots,\min\{R_{\rm S},L\},
$$

若 $R_{\rm S}>|\mathcal L|$，其余列可由正交补或随机单位向量初始化。

将全部初始列组成

$$
\overline{\mathbf Q}.
$$

### 21.3 总发射功率缩放

定义

$$
\chi_0
=
\min\left\{
1,
\frac{P_{\rm D}^{\max}}{\|\overline{\mathbf Q}\|_F^2}
\right\}.
$$

取

$$
\boxed{
\mathbf Q^{(0)}
=
\sqrt{\chi_0}\,\overline{\mathbf Q}
},
$$

即可保证

$$
\|\mathbf Q^{(0)}\|_F^2\le P_{\rm D}^{\max}.
$$

上行幅度可取

$$
q_j^{(0)}
=
\kappa_{\rm U}
\sqrt{P_{{\rm U},j}^{\max}},
\qquad
0<\kappa_{\rm U}<1.
$$

初始化不再计算残余 SI 功率，也不需要为 SI 阈值预留功率裕量。

---

## 22. 完整算法伪代码

### 算法 1：无 SI 约束的 FP－拉格朗日内层与 FP-aware RCG 外层联合优化

本文 Proposed 对应实现中的 `position_update_mode="fp_aware"` 和 `fp_aware_candidate_iters=2`；保留的 `original` 默认分支仅用于向后兼容与对照，不代表本文 Proposed 采用旧的固定资源 Armijo。

**输入：** 用户与目标位置、波长、ToMA 几何参数、上下行功率预算、残余 SI 抑制系数 $\rho_{\rm SI}$、性能权重和收敛容差。

1. 生成多组满足缆绳定长和防碰撞约束的初始 APV；
2. 对每组初始 APV：
   1. 根据第 21 节初始化 $\mathbf Q$ 和 $\{q_j\}$，仅执行总发射功率缩放；
   2. 重复外层迭代：
      1. **固定 APV，执行内层资源优化：**
         1. 更新全部 $\{\mathbf u_l\}$；
         2. 更新全部 $\{\mathbf b_j\}$；
         3. 更新 LDT 辅助变量 $\{\eta_k^{\rm D}\}$、$\{\eta_j^{\rm U}\}$ 和 $\{\eta_l^{\rm S}\}$；
         4. 更新 QT 辅助变量 $\{\phi_k^{\rm D}\}$、$\{\phi_j^{\rm U}\}$ 和 $\{\boldsymbol\phi_l^{\rm S}\}$；
         5. 构造 $\mathbf A_Q$ 和 $\mathbf B_Q$，通过 $\lambda_{\rm P}$ 更新 $\mathbf Q$；
         6. 通过分段闭式解更新 $\{q_j\}$；
         7. 重算真实 SINR 和真实 WSR；
         8. 直至内层收敛；
      2. **按顺序更新发射 ToMA 位置：** 对每个 $m\in\mathcal M_{\rm T}$：
         1. 固定该端点更新开始时的当前资源，计算中央差分梯度；
         2. 执行黎曼梯度投影和 PR+ RCG 方向计算，并初始化回溯步长；
         3. 通过球面回缩生成候选位置，并检查缆绳定长和防碰撞条件；
         4. 对每个几何可行候选点，重建候选几何和信道，以当前资源为 warm start 执行最多 2 次内层 FP 更新；
         5. 用候选位置与候选资源的联合真实 WSR 检查 FP-aware Armijo 条件；若失败则缩小步长并继续回溯；
         6. 若验收通过，同步接受候选位置和候选资源，并将二者用于下一端点更新；
      3. **按顺序更新接收 ToMA 位置：** 对每个 $m\in\mathcal M_{\rm R}$ 执行相同的固定当前资源梯度与 FP-aware 候选点评价；
      4. 重算全部通信信道、目标回波信道、等效残余 SI 信道、真实 SINR 和真实 WSR；
   3. 若
$$
      \frac{
      |F^{(t+1)}-F^{(t)}|
      }{
      \max\{1,|F^{(t)}|\}
      }
      \le\varepsilon_{\rm out}
$$
      连续两次成立，则停止；
   4. 验证最终总发射功率、上行功率、缆绳定长、防碰撞条件和近场适用范围；
3. 在全部多起点结果中输出真实 WSR 最大的可行解。

算法中不存在残余 SI 约束、SI 拉格朗日乘子、SI 候选筛选或最终 SI 合格判定。

---

## 23. 收敛性说明

固定 APV 时，LDT 和 QT 在辅助变量取最优值时与原加权和速率目标等价。固定其他变量后：

- 感知和上行接收波束分别由广义特征向量或 MMSE 方向更新；
- LDT 与 QT 辅助变量具有闭式最优解；
- 发射矩阵子问题为带总功率球约束的凹二次最大化问题，可由一个拉格朗日乘子精确求解；
- 每个上行幅度子问题为区间上的凹二次最大化问题，具有分段闭式解。

因此，在标准连续性、有界性和各分块精确更新条件下，固定 APV 的内层 FP 迭代使目标值非下降，并收敛到资源优化子问题的一个局部驻点。

外层位置更新采用球面回缩保持缆绳定长，并通过防碰撞检查保持几何可行。每一个被接受的 FP-aware 候选点均满足第 20 节定义的充分增益条件，且候选位置与候选资源同步接受；实际实现还检查完整联合 WSR 在数值容差内非下降。因此，算法生成的联合 WSR 序列满足

$$
F(\boldsymbol\Xi^{(t+1)})
\ge
F(\boldsymbol\Xi^{(t)}),
$$

且由于可行功率域有界、端点位于紧致球面上，加权和速率序列有上界，从而可在目标值意义下收敛。实现中以连续两次外层相对 WSR 变化不超过 $\varepsilon_{\rm out}$ 作为原始停止条件。

由于位置梯度仍是固定当前资源的 $\operatorname{grad}_{C}F(C_t,R_t)$，而候选点评价只执行最多 2 次 warm-start FP，以上论述不是严格约化目标 RCG 的收敛证明。外层相对 WSR 停止条件本身也不蕴含 $\|\operatorname{grad}\Phi(C)\|=0$。鉴于 (P1) 仍包含分式耦合、球面流形和非凸防碰撞条件，本文不声称算法收敛到全局最优或严格约化目标的一阶驻点；最终结果应表述为由有限差分步长、FP 截断和停止容差共同决定的数值稳定解。

---

## 24. 计算复杂度

### 24.1 接收波束更新

每轮接收波束更新的主要代价约为

$$
\mathcal O
\left(
(|\mathcal J|+|\mathcal L|)
N_{\rm R}^3
\right).
$$

### 24.2 发射矩阵更新

构造 $\mathbf A_Q$ 后，一次 $N_{\rm T}\times N_{\rm T}$ 矩阵分解的复杂度约为

$$
\mathcal O(N_{\rm T}^3).
$$

对不同的 $\lambda_{\rm P}$ 可复用特征分解或 Cholesky 结构，二分搜索的附加代价主要与

$$
\mathcal O(I_\lambda N_{\rm T}^2D)
$$

同阶，其中 $I_\lambda$ 为二分次数。

### 24.3 上行幅度更新

计算全部 $\{q_j\}$ 的主要复杂度约为

$$
\mathcal O
\left(
|\mathcal J|^2N_{\rm R}
+
|\mathcal J||\mathcal L|N_{\rm R}
\right).
$$

### 24.4 外层位置更新

每个子无人机的三维中央差分约需 6 次真实目标函数评估。若一次评估复杂度为 $C_F$，则一轮全部位置梯度的主要复杂度约为

$$
\mathcal O(6MC_F),
$$

另加 Armijo 回溯产生的候选点评价。相比固定资源 Armijo，每个几何可行候选位置除重建几何、信道和检查防碰撞条件外，还要以当前资源为 warm start 执行最多 2 次内层 FP 更新，再计算联合真实 WSR。因此，FP-aware-2 增加了单次 line-search 候选点评价成本，回溯候选越多，附加 FP 开销越大。

与原 SCA-SDR 方法相比，本算法不需要：

- 通用 SDP 内点法；
- $\{\mathbf W_k\}$ 的半正定变量；
- 通信协方差秩一惩罚与恢复；
- 残余 SI 约束的对偶变量和候选解筛选。

主要计算由矩阵分解、线性方程求解、标量二分、位置梯度目标函数评估以及候选点 warm-start FP 构成。FP-aware-2 的优势不在于降低单次或总计算复杂度，而在于更充分地反映位置变化后的资源适配效果，从而可能减少 Joint AO 轮数并改善最终联合 WSR；因此存在“额外候选 FP 开销”与“更少 Joint AO 轮数及更好收敛性能”之间的折中。删除 SI 检测仍使验收逻辑更简洁，但不抵消上述 FP-aware 候选更新开销。

---

## 25. 感知波束数与模型普遍性

### 25.1 单感知波束

若

$$
R_{\rm S}=1,
$$

则

$$
\mathbf S=\mathbf v_1\mathbf v_1^H,
$$

总感知协方差至多秩一，复杂度最低，但多目标感知自由度受到较强限制。

### 25.2 有限多波束

若

$$
1<R_{\rm S}<N_{\rm T},
$$

则

$$
\operatorname{rank}(\mathbf S)\le R_{\rm S},
$$

属于低秩感知设计，可在性能和复杂度之间折中。

### 25.3 完全一般的 PSD 表示

若

$$
R_{\rm S}=N_{\rm T},
$$

任意

$$
\mathbf S\succeq0
$$

均可通过 Cholesky 分解或特征值分解写成

$$
\mathbf S=\mathbf V\mathbf V^H.
$$

因此，“波束因子形式”本身不必损失一般性；真正缩小可行域的是对 $R_{\rm S}$ 的低秩限制。

---

## 26. 数值仿真与基线设置

建议至少设置以下基线：

1. 原 SCA-SDR/SDP 内层算法；
2. 本文无 SI 约束的 FP－拉格朗日多波束算法；
3. 固定 ToMA 位置；
4. 随机几何可行 ToMA 位置；
5. 单感知波束 $R_{\rm S}=1$；
6. 多感知波束 $R_{\rm S}=|\mathcal L|$；
7. 完整因子 $R_{\rm S}=N_{\rm T}$；
8. 理想自干扰消除基线 $\rho_{\rm SI}=0$；
9. 不同残余 SI 抑制水平 $\rho_{\rm SI}$；
10. 仅通信与仅感知方案。

建议报告：

- 系统加权和速率；
- 下行、上行和感知分项速率；
- 不同 $\rho_{\rm SI}$ 下的性能变化；
- 内层和外层收敛曲线；
- 内外层迭代次数与运行时间；
- 不同 $R_{\rm S}$ 下的性能—复杂度关系；
- 固定位置、随机位置与优化位置的性能差异；
- 多起点结果的最佳值、均值和标准差。

由于模型中已彻底删除残余 SI 约束，仿真不再统计 SI 阈值违约率、候选解 SI 拒绝次数或所谓“安全可行率”。残余 SI 的作用应通过 $\rho_{\rm SI}$ 对上行、感知和系统总性能的影响来展示。

---

## 27. 可直接用于论文的系统与算法描述

> 本文研究一种单套 ToMA 辅助的全双工近场 ISAC 系统。主无人机固定于全局坐标原点，全部发射和接收拖曳缆绳共用同一物理锚点，偶数个拖曳子无人机被固定等分为发射组和接收组。下行、上行和目标回波采用纯 LoS USW 信道，保留逐阵元精确球面波距离相位；共锚点近距离直接 SI 则逐收发阵元对保留自由空间距离幅度衰减和球面波相位，并通过综合残余系数形成统一的等效残余 SI 信道。为避免多个不可辨识感知协方差造成变量冗余，将总感知协方差写成多波束因子形式 $\mathbf S=\mathbf V\mathbf V^H$，并将通信波束和感知波束合并为发射矩阵 $\mathbf Q=[\mathbf W,\mathbf V]$。目标回波和残余 SI 在接收信号层面仍然叠加，但在 SINR 的二阶统计中作为互不相关分量分别累加协方差，不保留 echo--echo 或 echo--RSI 交叉项。统一的等效残余 SI 继续进入上行与感知 SINR 分母，以反映全双工收发同时工作造成的性能损失，但系统不设置任何残余 SI 功率阈值、显式约束或候选解检测条件。系统联合优化通信与感知波束、上行功率、接收波束以及收发 ToMA 三维端点位置，以最大化上下行通信和感知速率的加权和。固定位置时，采用最大 SINR 接收器、拉格朗日对偶变换、二次变换及总功率拉格朗日乘子，获得发射矩阵和上行功率幅度的闭式或一维搜索更新；位置梯度仍在当前资源固定条件下计算，而每个几何可行候选位置均以当前资源为 warm start 执行最多 2 次内层 FP 更新，再按 FP-aware Armijo 条件验收，并同步接受候选位置与候选资源。该方法避免通用 SDP、通信协方差秩一恢复、SI 对偶变量和 SI 候选筛选，形成一种 FP－拉格朗日资源优化与 FP-aware 球面黎曼位置更新交替进行的联合优化算法。

---

## 28. 模型边界

本模型采用固定主无人机、公共锚点、固定半数收发分组、确定性纯 LoS 窄带 CSI、多波束感知因子化和显式忽略 UL→DL 干扰的假设。暂不考虑主无人机轨迹、姿态变化、NLoS 多径、缆绳弯曲、宽带波束偏斜、位置误差、子无人机功能切换、运动能耗、重构时延、阵元互耦和硬件非线性。

残余 SI 仅作为基站接收性能中的干扰项，不构成优化约束或检测条件。因此，本模型不能用于证明逐 RF 前端功率一定低于 LNA、混频器或 ADC 的饱和阈值，而是隐含假设被动隔离及模拟/数字消除能力足以维持接收机正常工作。该简化适合研究 ToMA 位置、通信感知波束和上行功率对系统加权和速率的影响。

---

## 29. 参考论文及继承关系

1. J. Ding, Z. Zhou, X. Shao, B. Jiao, and R. Zhang, “Movable Antenna-Aided Near-Field Integrated Sensing and Communication,” *IEEE Transactions on Wireless Communications*, vol. 25, pp. 493–508, 2026.  
   继承全双工近场 ISAC、上下行通信与目标回波 SINR、加权和速率以及交替优化框架。

2. L. Zhu, H. Mao, W. Ma, Z. Xiao, J. Zhang, and R. Zhang, “Towed Movable Antenna (ToMA) Array for Ultra Secure Airborne Communications,” *IEEE Journal on Selected Areas in Communications*, vol. 44, 2026.  
   继承缆绳端点决定整根拖曳阵列、球面流形位置约束、子无人机防碰撞和黎曼位置优化思想。

3. B. Zhou, H. Gao, Z. Wei, X. Li, J. Wang, Y. Zhuang, and W. Wang, “Self-Interference-Alleviated Multi-Beam Steering for On-Demand Sensing and Communication Performance Tradeoff of Full-Duplex ISAC,” *IEEE Transactions on Wireless Communications*, vol. 25, pp. 177–194, 2026.  
   借鉴 LDT、分式规划、二次变换和拉格朗日闭式波束更新思想。本文保留残余 SI 对上行和感知性能的影响，但不设置任何残余 SI 阈值约束、SI 对偶变量或候选解检测机制。
