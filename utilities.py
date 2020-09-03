# A file for utility functions used to solve data science challenge for TagUp.INC
# Omkar Pawar: Master of Data Science.

import seaborn as sns
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#********************************************************************************************#
#Plot the histogram of the data
def show_histogram(data):
    '''Params: data: A dataframe of machine readings with 4 columns
       Details: Plots histogram showing distribution of each variable in a single plot'''
    
    bins = 20
    
    plt.hist(data["0"],bins,alpha = 0.5,label = "0")
    plt.hist(data["1"],bins,alpha = 0.5,label = "1")
    plt.hist(data["2"],bins,alpha = 0.5,label = "2")
    plt.hist(data["3"],bins,alpha = 0.5,label = "3")
    
    plt.legend(loc = "upper right")
    plt.xlabel("Readings")
    plt.ylabel("Frequency")
    plt.title("Histogram for Machine")
    plt.show()
#********************************************************************************************#
# Function to replace outliers with median
# Outliers are those that lie 2 stddev away from median
def replace_with_median(dataframe):
    '''Params: data: A dataframe of machine readings with 4 columns
       Details: Replace outliers with median.
       Return: Cleaned dataframe'''
    
    cols = dataframe.columns
    for i in cols:
        median = dataframe[i].median()
        std = dataframe[i].std()
        outliers = (dataframe[i] - median).abs() > (2*std)
        dataframe[i][outliers] = np.nan
        dataframe[i].fillna(median, inplace=True)
        
    return dataframe
#********************************************************************************************#

# Function to remove outliers from the data
def remove_outliers(df):
    '''Params: data: A dataframe of machine readings with 4 columns
       Details: Remove outliers from the dataframe.
       Return: Cleaned dataframe'''
    
    df_new = df[((df["0"] < 100) & (df["0"] > -100) &
                 (df["1"] < 100) & (df["1"] > -100) &
                 (df["2"] < 100) & (df["2"] > -100) &
                 (df["3"] < 100) & (df["3"] > -100))]
    
    return df_new
#********************************************************************************************#
#Function to draw the scatterplot of each variable of the given machine data.
def scatter(data,machine_no = 0):
    '''Params: data: A dataframe of machine readings with 4 columns
       Details: Plots scatterplot showing distribution of each variable in a single plot'''
    
    fig = plt.figure(figsize=(15,3))
    ax = fig.add_subplot(111)
    
    ax.scatter(range(len(data)),data["0"],alpha = 0.3, label = "0")
    ax.scatter(range(len(data)),data["1"],alpha = 0.3, label = "1")
    ax.scatter(range(len(data)),data["2"],alpha = 0.3, label = "2")
    ax.scatter(range(len(data)),data["3"],alpha = 0.3, label = "3")
    
    plt.xlabel("Timeline")
    plt.ylabel("Reading")
    plt.title("ScatterPlot for each variable of the machine {}".format(machine_no))
    plt.legend(loc = "upper right")
#********************************************************************************************#
# Detect Faulty regions according to each column
# Use confidence bands with top few rows
def Chauvenet(df,colname):
    '''Params: data: A dataframe of machine readings with 4 columns
               colname: Column name on which the faults are to be detected.
       Details: Chauvenet’s Criterion
                Detects faulty region in given column. Calculates mean and standard deviation
                of first 30 values and defines confidence bands. 
       Return: Point of transition from normal to faulty state.'''

    begin_with_data = 30

    mean = df[colname].head(begin_with_data).mean()
    std = df[colname].head(begin_with_data).std()
    upper = mean + 2*std
    lower = mean - 2*std
    #upper,lower
    counter = 0
    for i in range(len(df)):
        if (df[colname][i]>upper) | (df[colname][i]<lower):
            #print("Fault Detected at {}".format(i))
            counter+=1

            if counter > 3:
                #print("Faulty region")
                #print(i)
                break

    #scatter(df)
    #plt.plot([0,3000],[upper,upper])
    #plt.plot([0,3000],[lower,lower])
    #plt.plot([i,i],[-100,100])

    return i
#********************************************************************************************#
# Using rolling statistics to find faulty regions.
def rolling_statistic_detection(df,colname):
    '''Params: data: A dataframe of machine readings with 4 columns
               colname: Column name on which the faults are to be detected.
       Details: Rolling Statistics Approach
                Detects faulty region in given column. Calculates rolling mean, standard deviation
                of window size 35 and defines confidence bands accordingly. 
       Return: Point of transition from normal to faulty state.'''

    window_size = 35

    df["Rolling_Mean_"+colname] = df[colname].rolling(window = window_size).mean()
    df["Rolling_STD_"+colname] = 2*(df[colname].rolling(window = window_size).std())

    df["upper"] = df["Rolling_Mean_"+colname] + df["Rolling_STD_"+colname]
    df["lower"] = df["Rolling_Mean_"+colname] - df["Rolling_STD_"+colname]

    counter = 0
    for i in range(len(df)):

        if (df[colname][i]>df["upper"][i]) | (df[colname][i]<df["lower"][i]):
            #print("Fault Detected at {}".format(i))
            counter+=1

            if counter > 2:
                #print("Faulty region")
                #print(i)
                break

    #scatter(df)
    #plt.plot(range(len(df)),df["upper"])
    #plt.plot(range(len(df)),df["lower"])
    #plt.plot([i,i],[-100,100])

    return i
#********************************************************************************************#

#Define fault beginning regionsdef define_fault_region_start(df,method):
def define_fault_region(df,method,machine_no = 0):
    '''Params: data: A dataframe of machine readings with 4 columns
               method: Function to be used to get faulty region. "Chauvenet" and "rolling_statistic_detection"
               machine_no: Machine number for which dataframe is passed and region is defined.
       Details: Finds faulty regions in each column. Find the time when machine entered in faulty state.
                Finds Time to Failure
       Return: Time to Failure for the machine.'''
    #Two methods are implemented. Chauvenet and rolling_statistic_detection
    fault = []
    fault.append(method(df,"0"))
    fault.append(method(df,"1"))
    fault.append(method(df,"2"))
    fault.append(method(df,"3"))
    fault.sort()

    print("\nMachine {} entered faulty state at {}".format( machine_no, (df.index[fault[1]])) )

    faulty_region_idx = fault[1]
    fail_point_idx = failure_point(df)

    TTF = time_to_failure(df.index[faulty_region_idx], df.index[fail_point_idx])

    scatter(df,machine_no)
    plt.plot([faulty_region_idx,faulty_region_idx],[-100,100],label = "Faulty", color = "yellow")
    plt.plot([fail_point_idx,fail_point_idx],[-100,100],label = "Failed", color = "red")
    plt.legend(loc = 'upper right')
    print("Time to failure was {}".format(TTF))
    return TTF
#********************************************************************************************#
# Find the failure point
def failure_point(df):
    '''Params: df: A dataframe of machine readings with 4 columns
       Details: Find the failure point of the machine. The region where all observations are close to zero.
       Return: Point of machine failure.'''
    df = df.round(1)
    count = 0

    for i in range(len(df)):
        if((df["0"][i] == 0) &
           (df["1"][i] == 0) &
           (df["2"][i] == 0) &
           (df["3"][i] == 0) ):
            if count > 1:
                return i
            count+=1
#********************************************************************************************#
# Calcualte time to failure for the machine when it transitions to faulty and then failure
def time_to_failure(fault_point, failure_point):
    '''Params: fault_point: The point at which machine enters faulty region. Index number of fault point
               failure_point: The point at which machine fails. Index number of failure point
       Details: Calculate TTF by finding the differnce between both values.
       Return: Time to Failure for the machine.'''
    
    fault_time = pd.to_datetime(fault_point, format='%Y-%m-%d %H:%M:%S')
    fail_time = pd.to_datetime(failure_point, format='%Y-%m-%d %H:%M:%S')
    return (fail_time-fault_time)
#********************************************************************************************#
def average_TTF(TTF):
    '''Params: TTF: List of TTF for each machine (Machine_0 t0 Machine_19)
       Details: Calculate average TTF by removing negative entries.'''
    # Now that we have the values of Time to Failure for each machine, we will calculate the average time to failure
    l = []
    for i in TTF:
        l.append(int(str(i).split(" ")[0]))

    num_list = [item for item in l if item >= 0]
    Avg_TTF = sum(num_list)/len(num_list)
    print("\nAverage Time to failure considering all the machines is {} days".format(Avg_TTF))
#********************************************************************************************#
def performance(TTF):
    '''Params: TTF: List of TTF for each machine (Machine_0 to Machine_19)
       Details: Calculate performance of the model'''
    # Now that we have the values of Time to Failure for each machine, we will calculate the average time to failure
    l = []
    for i in TTF:
        l.append(int(str(i).split(" ")[0]))

    num_list = [item for item in l if item >= 0]
    accuracy = len(num_list)/len(TTF)
    print("\nWe were able to find faulty transition in {} machines".format(len(num_list)))
    print("Model performance is {}".format(accuracy))
#********************************************************************************************#      
         
