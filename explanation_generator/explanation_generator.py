#==============================================================================
import os
from pathlib import Path
import pickle
import numpy as np
from scipy.special import gammainc
import matplotlib.pyplot as plt
import random
import scipy.stats as stats

# Helper imports for the policy, descriptors and dimensions
from .utils import Concept, Dimension, normalize

#=============================================================================
class ExplanationGenerator:
    def __init__(self, actions, policy, classifier_storage_path):
        self.classifier_storage_path = classifier_storage_path
        # Construct the representation of the state space containing the
        # dimensions and their descriptors. dimensions is a list of dimensions
        # (each of them consisting of multiple descriptors)
        self.dimensions =  self.generate_state_space()
        self.actions = actions # A list containing the id and natural language representation of each action
        # Generate New Policy
        self.policy = policy
        
    def plot_policy(self, ax):
        """
        This function is used to plot the policy
        """
        x = np.linspace(0,1,200)
        y = []
        for i in range(0,len(x)):
            if self.policy.getPolicyY(x[i]) > self.dimensions[0].max_range:
                y.append(self.dimensions[0].max_range)
            else:
                y.append(self.policy.getPolicyY(x[i]))
        ax.plot(x,y)
    def plot2D(self, state, d1, d2):
        """
        This function plots 2 dimensions of the state space, the state, concept 
        boundaries and the policy
        """
        radius = 0.8 * self.dimensions[d1].max_range/len(self.dimensions[d1].descriptors)
        fig, ax = plt.subplots(1, 1)
        ax.set_autoscale_on = False
        # Plot the action decision boundary/policy
        self.plot_policy(ax)
        # Plot state and sample radius
        circle = plt.Circle((state[0],state[1]), radius, color='black', fill=False)
        ax.add_artist(circle)
        s = plt.Circle((state[0], state[1]),0.01, color='r')
        ax.add_artist(s)
        # horizontal descriptors
        for i in range(0, len(self.dimensions[d2].thresholds)):
            ax.plot((0, self.dimensions[d2].max_range),(self.dimensions[d2].thresholds[i], self.dimensions[d2].thresholds[i]), '-r')
            #ax.plot((0, self.dimensions[d1].max_range),(self.dimensions[d1].thresholds[i],self.dimensions[d1].thresholds[i]), '-r')
        # vertical descriptors
        x = np.linspace(0, self.dimensions[d1].max_range,101)
        for i in range(0, len(self.dimensions[d1].thresholds)):
            y = np.full((np.shape(x)), self.dimensions[d1].thresholds[i])
            ax.plot(y,x,'-r')
            
        plt.show()
    def generate_state_space(self):
        """
        This funciton generates a state space consisting of a number of
        dimensions which consist of descriptor. Each .pkl file in the subfolder 
        "classifiers" represents one dimension
        """
        # Iterate through all pkl files in the "classifiers" folder
        dimensions = []
        for filename in sorted(os.listdir(self.classifier_storage_path)):
            if filename.endswith(".pkl"):
                # For each of the pkl files/dimensions read the dimensions name,
                # descriptors, maximum/minimum range and natural language
                # representation from the pkl file
                descriptors= []
                fname = os.path.join(self.classifier_storage_path,str(filename))
                with open(str(fname), 'rb') as f:
                    #i = pickle.load(f)
                    name = pickle.load(f)
                    descriptors= pickle.load(f)
                    max_range = pickle.load(f)
                    min_range = pickle.load(f)
                    natural_language_representation = pickle.load(f)
                # Create new dimension with name, concept list, max/min range of
                # the dimension and the natural language representation
                dimensions.append(Dimension(name, descriptors, max_range, min_range, \
                                            natural_language_representation))
        return dimensions
    
    # Helper functions to generate explanations
    def sampleAround(self, radius, num_samples, center):    
        """
        This function generates samples in a hypersphere around a state
        radius: The radius of the hypersphere
        num_smaples: The amount of samples to generate
        center: The center of the hypersphere (the state we are sampling around)
        """
        r = radius
        ndim = center.size
        out = np.zeros((num_samples,ndim)) 
        for i in range(0, num_samples):
            x = np.random.normal(size=(1, ndim))
            ssq = np.sum(x**2,axis=1)
            fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
            frtiled = np.tile(fr.reshape(1,1),(1,ndim))
            p = center + np.multiply(x,frtiled)
            for j in range(0,ndim):
                out[i,j] = p[0,j]
        return out
    def sampleAlong(self, state, d, num_samples, random=False):
        """
        This function creates num_samples samples for one dimension d with all other dimensions fixed.
        """
        # Fix all other dimensions
        # Prepare the samples array
        samples = np.zeros((num_samples,len(self.dimensions)))
        # Create the non fixed array that will be representing the sampling line and fill it 
        # into the samples array
        if random:
            x = np.random.uniform(0, self.dimensions[d].max_range, num_samples)
        else:
            x = np.arange(self.dimensions[d].min_range, self.dimensions[d].max_range, (self.dimensions[d].max_range-self.dimensions[d].min_range)/num_samples)
        x = x.reshape((num_samples,1))
        samples[:,d] = x[:,0]

        # Fill the samples array with the other fixed dimensions
        for dim in range(0, len(self.dimensions)):
            if dim is not d:
                y = np.full((len(x),(len(self.dimensions)-1)),state[dim])
                samples[:,dim] = y[:,0]
        return samples
    def getNormalizedClassification(self, x_value, d, c):
        """
        This function returns a value v from the normalized classifiers
        x_value: The x value on the dimension d
        c: Which classifier to use to classify x_value
        """
        values = []
        for i in range(0,len(self.dimensions[d].descriptors)):
            values.append(self.dimensions[d].descriptors[i].normal_distribution.pdf(x_value))
        return normalize(values)[c]
        
        return out
    def isInConcept(self, sample):
        """
        Return an array with the maximum likelihood concept for each dimension. Can be used to querry a state or a number of 
        samples.
        """
        out = []
        for i in range(0, len(sample)):
            temp = []
            for j in range(0, len(self.dimensions[i].descriptors)):
                temp.append(self.dimensions[i].descriptors[j].normal_distribution.pdf(sample[i]))
            #print(temp)
            if len(temp) > 0:
                out.append(temp.index(max(temp)))
            else:
                out.append(0)
        return out
    def generate_natural_language_explanation(self, state, actions, threshold, entropy_threshold, N,rover, verbose=False):
        """
        This function generates a natural language explanation.
        
        It uses the generate_explanation function to retrieve stability, describability, relevance and consistency for the given state and actions.
        
        Input:
        state, list: The state that has to be explained
        actions, list: A list of actions that exist in the state space
        threshold, float: The threshold for the combined stability, describability and consistency
        threshold_relevance, float: The threshold for the relevance
        
        Output
        e, list: A list containing an explanation for each dimension. If the dimension fails the
            exlanation quality measures it will instead return which measure fails for the given
            dimension
        explainable_dimensions, int: A list of dimension that can be used to explain
        """
        stability, describability, relevance, relevance_threshold, consistency = self.generate_explanation(state, actions, entropy_threshold)
        state_action = self.policy.getAction(state)
        if verbose:
            print("stability: {}".format(stability))
            print("describability: {}".format(describability))
            print("relevance: {}".format(relevance))
            print("relevance_threshold: {}".format(relevance_threshold))
            print("consistency: {}".format(consistency))
            
        
        # Quality measure for each dimension depending on stability, describability and consistency
        local_measure = stability * describability
        global_measure = consistency*local_measure*relevance
        used_to_explain = [] # This list stores which dimensions can be used for an explanation
        
        # These listst are used to track how often a certain measure fails to generate an explanation in cases where no 
        # dimension can be used to explain
        stables = []
        describables = []
        consistencies = []
        relevances = []
        
        # Check Stability, describability and consistency
        for d in range(0, len(self.dimensions)):
            stable = True
            describable = True
            consistent = True
            relevant = True
            dimension_name = self.dimensions[d].name[rover]
            explanation_concept = self.isInConcept(state)[d] # Get the maximum likelihood concept
            concept_name = self.dimensions[d].descriptors[explanation_concept].natural_language_representation
            if stability[d] < threshold[0]:
                stable = False
            if describability[d] < threshold[1]:
                describable = False
            if consistency[d] < threshold[2]:
                consistent = False
            # Check relevance
            if relevance[d] < relevance_threshold:
                relevant = False
            stables.append(stable)
            describables.append(describable)
            consistencies.append(consistent)
            relevances.append(relevant)
            #print("dim: {}, stable: {}, describable: {}, consistent: {}, relevant: {}".format(d, stability[d], describability[d], consistency[d], relevance[d]))    
            #print("dim: {}, stable: {}, describable: {}, consistent: {}, relevant: {}".format(d, stable, describable, consistent, relevant))
            #print("=")
            if stable and describable and consistent and relevant:
                used_to_explain.append([global_measure[d], dimension_name, concept_name])
            else:
                used_to_explain.append([0, dimension_name, concept_name])
        #print("used_to_explain: {}".format(used_to_explain))
        # Get the indices of dimensions that we use for explaining (the last N elemnts of this array are the largest global measures in ascending order)
        sorted_indices = (np.argsort(np.array(used_to_explain)[:,0])[-N:])[::-1]
        #print("sorted_indices: {}".format(sorted_indices[::-1]))
        #print("state: {}, action: {}".format(state, state_action))
        # Catch if the explanation is possible or not:
        explanation = ""
        numerical = []
        if len(sorted_indices) > 0:
            if used_to_explain[sorted_indices[0]][0] > 0:
                explanation = "I did {} because".format(self.actions[state_action].name[rover])
                numerical.append(state_action)
                explanation = explanation + " {} was {}".format(used_to_explain[sorted_indices[0]][1], used_to_explain[sorted_indices[0]][2])
                numerical.append([used_to_explain[sorted_indices[0]][1],explanation_concept])
                for j in range(1, len(sorted_indices)):
                    #print("used to explain: {}".format(used_to_explain[sorted_indices[j]][0]))
                    if used_to_explain[sorted_indices[j]][0] > 0:
                        explanation = explanation + " and {} was {}".format(used_to_explain[sorted_indices[j]][1], used_to_explain[sorted_indices[j]][2])
                        numerical.append([used_to_explain[sorted_indices[j]][1],explanation_concept])
        if explanation == "":
            # Count number of False for each measure
            nat_lang_reps = ["not stable", "not describable", "not consistent", "not relevant"]
            stables = sum(stables)
            describables = sum(describables)
            consistencies = sum(consistencies)
            relevances = sum(relevances)
            temp = [stables, describables, consistencies, relevances]
            
            # If explanation is not stable or describable: the action selection in this situation is not clear
            # If explanation is not consistent: i have no descriptors that only describes this action
            # If explanation is not relevant: none of the dimensions contributes to the action selection
            
            # Use the reason with the most failed measure first
            s = np.argsort(temp)[::-1]
            explanation = "I could not explain why i did {}".format(self.actions[state_action].name[rover])
            if len(s) > 0:
                if stables > 0:
                    explanation += " because the action selection in this situation is not clear"
                if describables > 0:
                    explanation += ", there are not enough descriptors to describe this situation"
                if consistencies > 0:
                    explanation += ", I have no descriptors that only describe this action"
                if relevances > 0:
                    explanation += ", some of the current parameters contribute to the action selection"
        return explanation, numerical
    def compute_relevance_threshold(self, number_actions, entropy_limit):
        """
        This functions computes a threshold for which a state is still leading to a relevant action.
        The goal is to generate a vector temp that leads to the lowest entropy that we still want to take into account
        when using the relevance as a mask.
        Example: entropy_limit = 0.85 -> temp = [0.85, 0.075, 0.075] -> relevance threshold = 0.5266
        input: action_number: number of possible actions
        """
        temp = [(1-entropy_limit)/(number_actions-1)] * number_actions
        temp[0] = entropy_limit
        #print("relevance threshold: {}".format(stats.entropy(temp)))
        return stats.entropy(temp)
    def compute_relevance(self, state, actions, d, samples,state_action):
        """
        This function computes the relevance of a dimension d. It fixes all other dimensions and varies the dimension to
        evaluate. Based on the ratio of samples with same action as the state we want to explain to all samples we compute
        the relevance
        param state: The state we want to explain
        param number_actions: 
        param state_action: The action of the state we want to explain
        """
        # Variables to compute the ratios
        a_o = 0
        a_s = 0
        p = [] # Vector that contains the relevance for all actions along the dimension d we sample from
        # Classify the samples depending on their action (count samples with same action
        # and different actions as the state
        #=====
        for a in actions:
            a_o = 0
            a_s = 0
            for i in range(0, np.shape(samples)[0]):
                sample_action = self.policy.getAction(samples[i])
                sam = samples[i][0:len(self.dimensions)]
                if a.id_number == sample_action:
                #if state_action == sample_action:
                    a_s += 1
                else:
                    a_o += 1
            p.append(a_s/(a_s+a_o))
        #print("p: {}, entropy: {}".format(p, stats.entropy(p)))
        return stats.entropy(p)
    def compute_consistency(self, state, d, state_concept, state_action, samples):
        """
        This function computes the consistency of a dimension d.
        param state: The state we want to explain
        param state_concept: The concept of the state we want to explain for dimension d
        param state_action: The action of the state we want to explain
        """
        # Variables to compute the ratios
        a_s_c_s = 0
        a_s_c_o = 0
        a_o_c_s = 0
        a_o_c_o = 0
        # Classify the samples depending on their action (count samples with same action
        # and different actions
        y = np.array([state[1]]*np.shape(samples)[0])
        for i in range(0, np.shape(samples)[0]):
            sample_action = self.policy.getAction(samples[i])
            sam = samples[i][0:len(self.dimensions)]
            if sample_action == state_action:
                a_s_c_s += self.getNormalizedClassification(sam[d],d,state_concept)
                a_s_c_o += 1-self.getNormalizedClassification(sam[d],d,state_concept)
            else:
                a_o_c_s += self.getNormalizedClassification(sam[d],d,state_concept)
                a_o_c_o += 1-self.getNormalizedClassification(sam[d],d,state_concept)
        c_s = a_s_c_s + a_o_c_s
        c_o = a_s_c_o + a_o_c_o
        return ((a_s_c_s)/(c_s))#a_s_c_s/1
    def compute_stability(self, state_action, samples):
        """
        This function computes the stability of an explanation for a state.
        param state_action: The action of the state we want to explain
        param samples: local samples around the state
        return: a measure for the statbility of the explanation for a given state action and samples around the state
        """
        a_s = 0
        a_o = 0
        for i in range(0, np.shape(samples)[0]):
            sam = samples[i][0:len(self.dimensions)]
            sample_action = self.policy.getAction(sam)
            if sample_action == state_action:
                a_s += 1
            else:
                a_o += 1
        return a_s/(a_s+a_o)
    def compute_describability(self, d, state_concept, state_action, samples):
        c_s = 0
        c_o = 0
        a_s_c_s = 0
        a_s_c_o = 0
        a_o_c_s = 0
        a_o_c_o = 0
        for i in range(0, np.shape(samples)[0]):
            sam = samples[i][0:len(self.dimensions)]
            sample_action = self.policy.getAction(sam)
            if sample_action == state_action:
                a_s_c_s += self.getNormalizedClassification(sam[d],d,state_concept)
                a_s_c_o += 1-self.getNormalizedClassification(sam[d],d,state_concept)
            else:
                a_o_c_s += self.getNormalizedClassification(sam[d],d,state_concept)
                a_o_c_o += 1-self.getNormalizedClassification(sam[d],d,state_concept)
        c_s = a_s_c_s + a_o_c_s
        c_o = a_s_c_o + a_o_c_o # = 1-cs
        return c_s/(c_s+c_o)
    def generate_explanation(self, state, actions,relevance_threshold, num_samples_local=500, num_samples_global=100):
        """
        This function generates explanation quality metrics for a state.
        param state: the state to explain
        param num_samples: the number of samples used for the spherical sampling
        return: a np array of stabilities, describabilities, relevances and consistencies for each dimension
        """
        
        # Output storages
        stabilities = []
        describabilities = []
        relevances = []
        consistencies = []
        number_actions = len(actions) # length of all possible actions
        relevance_threshold = self.compute_relevance_threshold(number_actions, relevance_threshold)
        
        # Get the action and the concept of the sate we want to explain
        state_action = self.policy.getAction(state)
        state_concept = self.isInConcept(state)
        # For all dimensions compute stability, describability, relevance, consitency and append it to a list
        for d in range(0, len(self.dimensions)):
            # Individual radius based on the number of descriptors in this dimension
            radius = 0.6 * self.dimensions[d].max_range/len(self.dimensions[d].descriptors)
            # Get num_samples_local samples around state with radius radius
            samples_local = self.sampleAround(radius, num_samples_local, np.array(state))
            # Get num_samples_global samples along dimension d with all other dimensions fixed
            samples_global = self.sampleAlong(state, d, num_samples_global)
            # Stability - "action change"
            stabilities.append(self.compute_stability(state_action, samples_local))
            # Describability - "concept change"
            describabilities.append(self.compute_describability(d, state_concept[d], state_action, samples_local))
            # Relevance of dimension d
            r = self.compute_relevance(state, actions, d, samples_global,state_action)
            relevances.append(r)
            # Consistency of dimension d
            consistencies.append(self.compute_consistency(state, d, state_concept[d], state_action, samples_global))
        #print("stab: {}, describ: {}, rel: {}, cons: {}".format(stabilities, describabilities, relevances, consistencies))
        return np.array(stabilities), np.array(describabilities), np.array(relevances), relevance_threshold, np.array(consistencies)
