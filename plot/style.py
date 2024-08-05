import matplotlib
import matplotlib.pyplot as plt

# Settings 
matplotlib.use("template")
fontsize = 22
plt.rc("font", family="serif", size=fontsize)
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)
plt.rc("axes", axisbelow=True)
plt.rc("legend", fontsize=fontsize, frameon=True, framealpha=1.0)

matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.rcParams["lines.linewidth"] = 1.5
matplotlib.rcParams["lines.markersize"] = 5

linestyles = ["solid", "dashdot", "dashed", "dotted", (0, (3, 1, 1, 1))]

colors = [ "#003366", "#e31b23", "#787878", "#1a9e00", "#03b5fc"] #RPTH Palette

markers = [ "o", "v", "s", "P", "X"] 