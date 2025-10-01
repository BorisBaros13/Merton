import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

# with open("entropy_diffs.pkl", "rb") as f:
#     bins_loaded = pickle.load(f)
numpy_bins = {key: df.to_numpy(dtype = float) for key, df in bins_loaded.items()}

# Combining your data:
data_group1 = [numpy_bins[key][:,0] for key, c in numpy_bins.items()]
# data_group2 = [numpy_bins[key][:,1] for key, c in numpy_bins.items()]
# data_group3 = [numpy_bins[key][:,2] for key, c in numpy_bins.items()]
  
colors = ["red", "blue", "orange"]
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'font.size': 14
})
 
data_groups = [data_group1]#, data_group2, data_group3]

#Labels for your data:
labels_list = [32, 64, 128]
width       = 1/len(labels_list)
xlocations  = [ x*((1+ len(data_groups))*width) for x in range(len(data_group1)) ]
symbol      = 'r+'
ymin        = min ( [ val  for dg in data_groups  for data in dg for val in data ] )
ymax        = max ( [ val  for dg in data_groups  for data in dg for val in data ])
import matplotlib.ticker as ticker
ax = plt.gca()
ax.set_ylim(ymin/2, ymax*2)
ax.set_yscale("log", base = 2)
ax.yaxis.set_minor_locator(ticker.LogLocator(base = 2.0, subs = (1.0, ), numticks = 100))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.grid(which = "major", color = "black", linestyle='dotted')
ax.grid(which = "minor", color = "gray", linestyle = "dotted")
ax.set_axisbelow(True)
plt.xlabel('Sample Size')
plt.ylabel('Generalisation Error Estimate')
#plt.title("Regularised Merton Problem, 50 Trials")
space = len(data_groups)/2
offset = len(data_groups)/2
# Offset the positions per group:
group_positions = []
for num, dg in enumerate(data_groups):    
    _off = (0 - space + (0.5+num))
    group_positions.append([x+_off*(width+0.01) for x in xlocations])
for dg, pos, c in zip(data_groups, group_positions, colors):
    boxes = ax.boxplot(dg, 
                sym=symbol,
                labels=labels_list,
                positions=pos, 
                widths=width, 
                boxprops=dict(facecolor=c),                      
                medianprops=dict(color='black'),
                patch_artist=True,
                )

line_xs = [xlocations[0], xlocations[-1]]
start_val = 2
line_ys = [start_val, start_val/(2**(len(labels_list)-1))]
ax.plot(line_xs, line_ys, color = "black", zorder = 0)
ax.set_xticks( xlocations )
ax.set_xticklabels( labels_list, rotation=0 )
import matplotlib.patches as mpatches
red = mpatches.Patch(color = "red", label = "10 Neurons")
blue = mpatches.Patch(color = "blue", label = "100 Neurons")
orange = mpatches.Patch(color = "orange", label = "1000 Neurons")
means = mpatches.Patch(color = "black", label = "Reference Line")
#plt.legend(handles = [red, blue, orange])
plt.legend(handles = [red, blue, orange, means])
plt.show()
