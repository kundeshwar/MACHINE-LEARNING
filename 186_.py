import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import seaborn as se

dataset = pd.read_csv(r"position.csv")
se.lmplot(x='level',y='salary',data=dataset)

mp.show()

