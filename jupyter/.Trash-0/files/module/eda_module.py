import matplotlib.pyplot as plt
import seaborn as sns

# Construction de graphique à barres interactive
def bar_plot(data, Variable):
    plt.rcParams["figure.figsize"] = (7, 4)
    sns.set_style("darkgrid")
    data[Variable].value_counts(normalize=True).plot(kind='bar', color='violet')
    plt.ylabel('proportion', size=14)
    plt.title('Distribution of ' + str(Variable), size=16)
    return plt.show()

# Construction d'histogrammes interactive
def hist_plot(data, Variable):
    plt.rcParams["figure.figsize"] = (7, 4)
    sns.set_style("darkgrid")
    sns.distplot(data[Variable].dropna(), kde=False)
    plt.title('Histogram of ' + str(Variable), size=16)
    return plt.show()

# Construction de boîtes à moustaches interactive
def box_plot(data, Variable):
    plt.rcParams["figure.figsize"] = (7, 4)
    sns.set_style("darkgrid")
    sns.boxplot(y=data[Variable].dropna(), color='yellow')
    plt.title('Boxplot of ' + str(Variable), size=16)
    return plt.show()
