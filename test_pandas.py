#!/usr/bin/python

import pandas as pd
import numpy as np

df=pd.DataFrame(index=range(5))
df['Val']=''
df.loc[[1,4],'Val']='bad'
df.loc[[0,3],'Val']='good'
print df
