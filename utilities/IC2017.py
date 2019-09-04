import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc

def shuffled(array):
    x = array.values.copy()
    np.random.shuffle(x)
    return x

def shuffled2(array):
    x = array.copy()
    np.random.shuffle(x)
    return x

def getBestTreshold(FPR, TPR, TH):
    """This function is for calculating the Mean Accuracy, given a ROC curve"""
    i_max = max(range(len(TPR)), key = lambda x : TPR[x] + 1 - FPR[x])
    mean_accuracy = (TPR[i_max] + 1 - FPR[i_max])/2
    return mean_accuracy, TH[i_max]

def getPerformance(true, prediction):
    try:
        
        fpr, tpr, th = roc_curve(true, prediction)
        score = auc(fpr, tpr)        
        if score < .5:
            fpr, tpr = tpr, fpr
            score = 1 - score
        mean_acc, bestTH = getBestTreshold(fpr, tpr, th)
        return score, mean_acc, bestTH, fpr, tpr
    
    except: # might happen when there is only one class
        
        return None, np.NaN, np.NaN, None, None

def splitByImages(data, test_size, repetitions):
    images = list(set(IMG.loc[data.index]))
    n = len(images)
    for _ in range(repetitions):
        np.random.shuffle(images)
        split_point = int(n * (1-test_size))
        yield images[:split_point], images[split_point:]