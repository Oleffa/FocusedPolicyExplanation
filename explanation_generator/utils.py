def normalize(raw):
    """
    This function normalizes a list
    Input:
        raw: List of float numbers that will be normalized
    Output:
        out: Normalized vector
    """
    if sum(raw) == 0:
        out = [0] * len(raw)
    else:
        out = [float(i)/sum(raw) for i in raw]
    return out
class Action:
    """
    This class defines an action. It stores the natural language represenation
    and the id of the action
    """
    def __init__(self, name, id_number):
        self.name = name
        self.id_number = id_number
class Performance:
    """
    This class is used to store the performance of a user
    """
    def __init__(self, user_id):
        self.user_id = user_id
        self.rover_1_actions = []
        self.rover_1_reaction_times = []
        self.rover_1_time = 0
        self.rover_1_learning_time = 0
        self.rover_1_evaluation_time = 0
        self.rover_2_actions = []
        self.rover_2_reaction_times = []
        self.rover_2_time = 0
        self.rover_2_learning_time = 0
        self.rover_2_evaluation_time = 0
        self.experiment_time = 0
class Concept:
    """    
    This class defines a concept it contains, its parameters and natural language description.
    A dimension consists of multiple descriptors
    """    
    def __init__(self, dimension, normal_distribution, natural_language_representation):
        self.dimension = dimension # The dimension this concept belongs to
        self.natural_language_representation = natural_language_representation # The natural language representation 
        self.normal_distribution = normal_distribution
class Scenario:
    def __init__(self, name, path, actions, explanations_good, explanations_random):
        self.name = name
        self.path = path
        self.actions = actions
        self.explanations_good = explanations_good
        self.explanations_random = explanations_random
class Dimension:
    """
    This class defines a dimensions, its parameters and natural language description
    """
    def __init__(self, name, descriptors, max_range, min_range, natural_language_representation):
        self.name = name # The name of the dimension "cart_velocity"
        self.descriptors= descriptors# A list of descriptors
        self.max_range = max_range # Max range of this dimension
        self.min_range = min_range # Min range of this dimension
        # All thresholds that appear for this certain dimensions. 
        # Thresholds are the points in this dimension where two neighbouring descriptors are equal.
        self.thresholds = self.calculate_thresholds()
        self.natural_language_representation = natural_language_representation
    def calculate_thresholds(self):
        """
        This function computes a list of points of a parameter where two neighbouring descriptors 
        are equal.
        Used for example to plot lines that separate descriptors.
        """
        thresholds = []
        for j in range(0, len(self.descriptors)-1):
            cur_concept_high = self.descriptors[j+1]
            cur_concept_low = self.descriptors[j]
            cur = (self.max_range/2)#/self.parameters[i].max_range
            init = cur
            #print("init: {}".format(init))
            for step in range(0, 10):    
                high = cur_concept_high.normal_distribution.pdf(cur)
                low = cur_concept_low.normal_distribution.pdf(cur)
                #print("high: {}, low: {}".format(high,low))
                if low == high:
                    continue
                if low < high:
                    cur = cur - init/2
                elif low > high:
                    cur = cur + init/2
                init = init/2
                #print("temp cur: {}".format(cur))
            #print("cur: " + str(cur))
            thresholds.append(cur)        
        return thresholds
