from imblearn.over_sampling import RandomOverSampler

class DatasetBalance:
    '''
    This class is used to balance the unbalanced dataset.
    How to use
    balance=DatasetBalance()
    balance.overs
    '''
    def __init__(self):
        pass
    def oversample1(self,x,y):
        '''
        This method uses RandomOverSampler to oversample the classes which have less number of samples
        '''
        ros = RandomOverSampler()
        x_ros, y_ros = ros.fit_sample(x, y)
        return x_ros,y_ros