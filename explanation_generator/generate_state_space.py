import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from scipy.stats import norm
from cycler import cycler
import os
# Custom utils function that contains the definitions for descriptors and dimensions
from explanation_generator import Concept

def normalize(raw):
    """
    This function normalizes
    Input:
        raw: vector of float numbers that will be normalized
    Output:
        out: normalized vector
    """
    if sum(raw) == 0:
        out = [0] * len(raw)
    else:
        out = [float(i)/sum(raw) for i in raw]
    return out

class DimensionGenerator:
    def __init__(self, classifier_storage_path):
        self.classifier_storage_path = classifier_storage_path
        #assert (os.path.isdir(self.classifier_storage_path)), "Error: Path to classifier storage directory not found. Hint: Make sure that the directory exsits!"
    def getIndexInLimits(self, x, limits):
        index = -1
        for i in range(0, len(limits)-1):
            if x > limits[i] and x <= limits[i+1]:
                index = i
        return index
    def getIndexInStorage(self, descriptor, storage):
        for i in range(0, len(storage[descriptor])):
            if storage[descriptor][i] <= 0:
                return i
        return len(storage[descriptor]-1)
    def clear_classifiers(self):
        if os.path.isdir(self.classifier_storage_path):
            filelist = [ f for f in os.listdir(self.classifier_storage_path) if f.endswith(".pkl") ]
            for f in filelist:
                os.remove(os.path.join(self.classifier_storage_path, f))
        else:
            os.makedirs(self.classifier_storage_path)
    def generate_dimension(self, name, descriptor_names, nlr, max_range, min_range, save, limits, duration, automated, normalized=True, savefig=False, plot=False):
        """
        This function automatically generates a dimension with the following dimension
        It simulates the learning from an expert and uses set limits to classify the samples
        name: The name of the dimension
        descriptor_names: the names of all descriptors of this dimension
        max_range, min_range: minimum and maximum range of the dimension
        save: Boolean if the dimension should be saved to a pkl file
        limits: A list of descriptor boundaries that are used to simulate the expert
        duration: Number of samples to label
        automated: Whether the datasets are labeled according to the limits given or by querying the user

        Plotting
        normalized: Whether the plot displays the normalized or the raw distributions
        savefig: Whether the plot geneated should be saved as a file named
        """
        num_descriptors= len(limits)-1
        storage = np.zeros((num_descriptors, duration))
        normal_distributions = []

        for i in range(0,duration):
            # Generate random entropy dataset
            x = round(random.uniform(0.0,1.0), 2)
            # Ask for label
            if automated == False:
                inp = int(input("Is this battery value: {} low(0), okay(1) or high(2)? ".format(x)))
            else:
                descriptor = self.getIndexInLimits(x,limits)
                storage[descriptor][self.getIndexInStorage(descriptor, storage)] = x
        for i in range(0, num_descriptors):
            sigma, mu = norm.fit(storage[i][:self.getIndexInStorage(i, storage)])
            normal_distribution = norm(sigma, mu)
            normal_distributions.append(normal_distribution)

        # Plotting
        if plot: 
            x = np.linspace(0, 1.0, 100)
            fig = plt.figure()
            if normalized == True:
                y = []
                for i in range(0, len(normal_distributions)):
                    y.append(normal_distributions[i].pdf(x))
                y = np.array(y)
                y_plot = np.zeros(np.shape(y))
                # Normalization
                for i in range (0, len(y[0])):
                    normalized = normalize(y[:,i])
                    for j in range(0, num_descriptors):
                        y_plot[j][i] = normalized[j]
                ax1= fig.gca()
                ax1.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y', 'm']) )
                for i in range(0, num_descriptors):
                    plt.plot(x, y_plot[i], lw = 4, alpha = 0.6, label='beta_pdf')
                fig.suptitle('Concept Classifiers for dimension: {}'.format(name), fontsize=14)
                plt.xlabel('Dimension', fontsize=12)
                plt.ylabel('Normalized Concept Strength', fontsize=12)
                if savefig:
                    fig.savefig('{}.jpg'.format(name))
            else:
                y = []
                for i in range(0, len(normal_distributions)):
                    y.append(normal_distributions[i].pdf(x))
                y = np.array(y)
                y_plot = y
                ax1= fig.gca()
                ax1.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y', 'm']) )
                for i in range(0, num_descriptors):
                    plt.plot(x, y_plot[i], lw = 4, alpha = 0.6, label='beta_pdf')
                fig.suptitle('Concept Classifiers for dimension: {}'.format(name), fontsize=14)
                plt.xlabel('Dimension', fontsize=12)
                plt.ylabel('Concept Strength', fontsize=12)
                if savefig:
                    fig.savefig('{}.jpg'.format(name))
            plt.show()

        # Build the Descriptors
        descriptors = []
        for i in range(0, num_descriptors):
            descriptors.append(Concept(name, normal_distributions[i], descriptor_names[i]))
        # Save the dimension to a file
        if save:
            file_dir = os.path.realpath(os.path.dirname(__file__))
            save_dir = os.path.join(self.classifier_storage_path+'/{}.pkl'.format(name))
            with open(save_dir, 'wb') as f:
                pickle.dump(name, f)
                pickle.dump(descriptors, f)
                pickle.dump(max_range, f)
                pickle.dump(min_range, f)
                pickle.dump(nlr, f)
