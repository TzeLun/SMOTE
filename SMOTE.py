# Written with Python ver. 3.6.8; NumPy ver. 1.19.1; Pandas ver. 1.1.1

import math as m
import random as rd
import numpy as np


# Details of argument:
# Format of arg : dataset : [[attr1, attr2, attr3, ... , class], [attr1, attr2, attr3, ... , class], .....]
# Dataset MUST contain like-classes, although users can modify code as a general synthesizer
# size must be in percentage, ie : 50%, 60%, 100%, 200%, 500%
# size = 100% means each data sample is synthesized once. Total synthetic data equals length(dataset)
# size = n*100% means each data sample is synthesized n times, for n >= 1
# for n < 1, only a portion of dataset is selected for one time synthesis
# default size : 100%
# k is the number of nearest neighbors for each data sample, default value, k = 5
# NOTE: No manual segregation of class label and attributes needed, algorithm pre-processes it automatically


def augment(dataset, size=100, k=5):
    # need to separate the class label and the numerical data -- pre-processing stage:
    data, c_label = [d[0:(len(d) - 1)] for d in dataset], dataset[0][3]
    n = size

    # if user chooses to synthesize less than the input data size:
    if size < 100:
        n = int((size/100)*len(data))  # determine the proportion of data samples for synthesis, no rounding needed
        ms = list()  # create an empty list to receive randomize samples of defined amount
        num = 0
        while num != n:
            ms.append(data.pop(rd.randrange(0, len(data))))  # append data samples and remove them to prevent duplicates
            num = num + 1
        data = ms
        n = 100  # set as 100% for one round of synthesis

    # Algorithm for SMOTE:
    synthetic = list()
    for i in data:
        mul = int(n / 100)  # integer multiple of the percentage chosen
        nn_array = list()  # array to store the indices of nearest neighbors for sample "i"
        nn_val = list()  # temporary list to store euclidean distance value for future comparison
        nn_index = list()  # temporary store indices for all neighbors of sample "i"

        # Computing euclidean distance between sample "i" and the rest of the data samples:
        for ind, dt in enumerate(data):
            if i == dt:
                continue  # do not compute nearest neighbors for same data
            else:
                nn_index.append(ind)
                nn_val.append(m.sqrt(sum(np.power(np.subtract(i, dt), 2))))

        # this if-condition removes bugs when user purposely define higher k than the available data samples:
        if k > len(data):
            k = len(data)

        # Keep the first few nearest neighbors based on k:
        count = 0
        while count < k:
            nn_array.append(nn_index[nn_val.index(min(nn_val))])  # Record the indices corresponding to nearest neighbor
            nn_index.remove(nn_index[nn_val.index(min(nn_val))])
            nn_val.remove(min(nn_val))
            count = count + 1

        # Synthesize data for data sample "i":
        while mul != 0:
            nn = rd.randrange(0, k)  # Randomly select one of the nearest neighbor, integer type
            diff = np.subtract(data[nn_array[nn]], i)  # Difference between two closely related samples
            var = rd.uniform(0, 1)  # Generate the variance multiplier, float type
            synthetic.append(list(np.add(i, var*np.array(diff))))  # Add sample i with variance vector
            mul = mul - 1

    # re-process the synthetic data by assigning class label and add it back to main dataset:
    sdata = [arr + [c_label] for arr in synthetic]
    return dataset + sdata
