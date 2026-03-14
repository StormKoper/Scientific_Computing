# imports
import matplotlib.pyplot as plt
import warnings

# setup global style of plotting
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] ='serif'
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14

# setup global style of warnings
def simple_format(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"
warnings.formatwarning = simple_format