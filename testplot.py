#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(-np.pi,np.pi,256,endpoint=True)
cos_wave=np.cos(x)
sin_wave=np.sin(x)

plt.plot(x,cos_wave,'r*-')
plt.plot(x,sin_wave,'bo-')
plt.show()

