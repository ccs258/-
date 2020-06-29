"""

"""

#使用numpy模块模拟生成5组，每组100个正态分布数据 正态分布函数参数依次是均值，标准差、数据的个数
import numpy as np
df = {'ctl':list(np.random.normal(10,5,100)),
      'treat1':list(np.random.normal(15,5,100)), \
      'treat2':list(np.random.normal(20,5,100)), \
      'treat3':list(np.random.normal(30,5,100)), \
      'treat4':list(np.random.normal(31,5,100))}
#组合成数据框
import pandas as pd
df = pd.DataFrame(df)
df.head()



df.boxplot(grid = False)
import matplotlib.pyplot as plt
plt.show()

#数据格式整理为一列为处理，一列为数值的形式
df_melt = df.melt()
df_melt.head()

df_melt.columns = ['Treat','Value']
df_melt.head()

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('Value~C(Treat)',data=df_melt).fit()
anova_table = anova_lm(model, typ = 2)
print(anova_table)

import seaborn as sns
sns.boxplot(x='Treat',y='Value',data = df_melt)


#参考https://zhuanlan.zhihu.com/p/91031244
