import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 200})

figSize = (6,6)#(12,8)
labelFS = 22
tickFS = 18
titleFS = 24
textFS = 16
legendFS = 16
tickwidth = 1.5

# Adjust axes
mpl.rcParams['axes.linewidth'] = 1.#2.5
mpl.rcParams['axes.axisbelow'] = 'line'

# Adjust Fonts
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['mathtext.fontset'] = "dejavusans"


mpl.rcParams['axes.titlesize'] = titleFS
mpl.rcParams['axes.labelsize'] = labelFS
mpl.rcParams['xtick.labelsize'] = tickFS
mpl.rcParams['ytick.labelsize'] = tickFS
mpl.rcParams['legend.fontsize'] = legendFS


# Adjust ticks
for a in ['x','y']:
    mpl.rcParams['{0}tick.major.size'.format(a)] = 10.0
    mpl.rcParams['{0}tick.minor.size'.format(a)] = 5.0

    mpl.rcParams['{0}tick.major.width'.format(a)] = tickwidth
    mpl.rcParams['{0}tick.minor.width'.format(a)] = tickwidth

    mpl.rcParams['{0}tick.direction'.format(a)] = 'in'
    mpl.rcParams['{0}tick.minor.visible'.format(a)] = True

mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# Adjust figure and subplots
mpl.rcParams['figure.figsize'] = figSize
mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.96
mpl.rcParams['figure.subplot.top'] = 0.96
