# SMOTE
A minority oversampling method for imbalance data set

## Brief Context on SMOTE
When dealing with large datasets, it is common to stumbled on uneven proportion of data classes. For example, a binary class dataset could contain 100,000 data samples but only 1,000 of them represents a particular data class whereas the rest are the opposite class. This creates a bias whereby the classifier favors the majority class. For example, when I was using CART or decision tree to classify breast cancer cells as benign or malignant with a class-imbalance dataset, I notice the classifier made a one-sided prediction on my validation data, eventhough the data contains two different classes. To solve this issue, one could undersample the majority class or oversample the minority class. Oversampling the minority class is like data augmentation, which in this case is done by synthesizing data using given data input, as proposed by [Chawla et. al 2002](https://arxiv.org/pdf/1106.1813.pdf), hence the name, Synthetic Minority Oversampling Technique (SMOTE).

## Rundown of Code

### Input Arguments for the Algorithm
**Three arguments :**
- **Dataset** : Format, [[attr1, attr2, attr3, ... , attrN, class], [attr1, attr2, attr3, ... , attrN, class], ... , [attr1, attr2, attr3, ... , attrN, class]], attr is the data attribute/variable. Need not separate data variables from its class label.
- **Amount of synthesis (size)** : This is in percentage format, ie : 20, 50, 100, 200, 400, 600. By default, this argument value os 100%, which means each data sample is synthesized once and the number of synthesized data is equal to length(input dataset). If the size is smaller than 100%, for instance 50%, this means only 50% of the input dataset is used to synthesize data once. For nx100%, each data sample is synthesized n times, hence size of synthetic data is n times bigger than input dataset.
- **Number of nearest neighbor, k** : Default k is 5.

### Preprocessing of Code
The written code below separates the data variables and the class label, hence no manual separation needed
``` Python
 # need to separate the class label and the numerical data -- pre-processing stage:
 data, c_label = [d[0:(len(d) - 1)] for d in dataset], dataset[0][3]
```

### For Size or Amount of Sampling < 100%
The code below reduces the number of data samples to the desired proportion used for synthesis by randomly selecting data from the input dataset. 
``` Python
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
```
### Evaluating Euclidean Distance for Each Data Sample
Nearest neighbors are evaluated using euclidean distance between each data sample, represented as "i", with the rest of the data sample. The nearest neighbor for two exact data samples are ignored. The suclidean distance and the indices of data sample corresponding to it is saved.
``` Python
 # Computing euclidean distance between sample "i" and the rest of the data samples:
        for ind, dt in enumerate(data):
            if i == dt:
                continue  # do not compute nearest neighbors for same data
            else:
                nn_index.append(ind)
                nn_val.append(m.sqrt(sum(np.power(np.subtract(i, dt), 2))))
```

### Choosing the "k" nearest neighbors
In case the user decides to set k larger than the length(input dataset), the code below helps remove this potential bug by clipping the value of k to length(input dataset) at max:
``` Python
 # this if-condition removes bugs when user purposely define higher k than the available data samples:
        if k > len(data):
            k = len(data)
```
Out of all the euclidean distance calculated, only the top "k" minimum distance value is counted as the nearest neighbors, hence the code below fetches the k nearest neighbors from the list:
``` Python
 # Keep the first few nearest neighbors based on k:
        count = 0
        while count < k:
            nn_array.append(nn_index[nn_val.index(min(nn_val))])  # Record the indices corresponding to nearest neighbor
            nn_index.remove(nn_index[nn_val.index(min(nn_val))])
            nn_val.remove(min(nn_val))
            count = count + 1
```

### Synthesizing Data
The code below shows how SMOTE synthesizes data by multiplying a random variance to the difference between the data sample and its random nearest neighbor:
``` Python
 # Synthesize data for data sample "i":
        while mul != 0:
            nn = rd.randrange(0, k)  # Randomly select one of the nearest neighbor, integer type
            diff = np.subtract(data[nn_array[nn]], i)  # Difference between two closely related samples
            var = rd.uniform(0, 1)  # Generate the variance multiplier, float type
            synthetic.append(list(np.add(i, var*np.array(diff))))  # Add sample i with variance vector
            mul = mul - 1
```

### SMOTE Algorithm as a Whole
What was shown is for one data sample. For all data samples from the dataset, below shows the whole algorithm:
``` Python
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
```

### Final Processing and Return Output
I wrote the algorithm in such a way that it also helps assigning the class label for the synthetic data and add it to the minority dataset, so no extra work needed, this is shown by a simple code below:
``` Python
 # re-process the synthetic data by assigning class label and add it back to main dataset:
    sdata = [arr + [c_label] for arr in synthetic]
    return dataset + sdata
```

## Algorithm Test
I created a simple binary dataset labeled as [sample.csv](https://github.com/TzeLun/SMOTE/blob/master/sample.csv) for anyone who wants to play with the algorithm. The dataset has 50 data of class 'A' and 25 data of class 'B'. I also include a Python Script called [sample.py](https://github.com/TzeLun/SMOTE/blob/master/sample.py) so you can start testing the algorithm promptly. The code is shown below:
``` Python
import pandas as pd
import SMOTE as sm

df = pd.read_csv('sample.csv', header=None)


# Simple pre-processing function to get the desired format for the dataset
def pre_processing(dataset):
    d = dataset.T
    return [list(d[i]) for i in d]


df = pre_processing(df)
minority = df[50:75]  # Use all 25 class 'B' data as the input dataset

# The SMOTE function is labelled as augment()
syn = sm.augment(minority, 50, 5)
print(syn)
print(len(syn))  # Expect 37, Number of data to be synthesize once : int(0.5*25) = 12. 
```

### Test Result
Below are just some result I would like to share to compare the synthetic data and their original counterparts:
``` Python
syn = sm.augment(minority, 100, 7)
# Comparing the first three data and their synthetic data
print(syn[0:3])
print(syn[25:28])
```
**Output :**
``` Python
# Original data
[[59.61642961, 8.595110319, 50.31237016, 'B'], [63.02832815, 8.697484548, 50.47710068, 'B'], [56.51687181, 8.185672353, 55.06152713, 'B']]

# Synthetic data
[[59.466073930579306, 8.543403844027365, 50.69430045815887, 'B'], [65.83293422613062, 8.208685479392788, 53.343374315915824, 'B'], [56.393525779457946, 8.27611909724483, 54.80230749914403, 'B']]
```
