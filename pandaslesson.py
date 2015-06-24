import pandas as pd
import numpy as np
import matplotlib as plt
df1 = pd.DataFrame(np.random.randn(6,3),index=list('abcdef'), columns=['a','b','c'])
df2 = pd.DataFrame(np.random.randn(6,3),index=list('cdefgh'),columns=['d','e','f'])
df3 = df1.copy()
df = pd.concat([df1,df2],axis = 1)
print df