import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def import_training_data():
    params = {}

    for i in range(1,6):
        params['X' + str(i)] = unpickle('data/data_batch_' + str(i))[b'data']
        params['Y' + str(i)] = unpickle('data/data_batch_' + str(i))[b'labels']

    params['X'] = np.append(params['X1'], params['X2'], axis=0)
    params['Y'] = np.append(params['Y1'], params['Y2'], axis=0)

    for i in range(3, 6):
        params['X'] = np.append(params['X'], params['X' + str(3)], axis=0)
        params['Y'] = np.append(params['Y'], params['Y' + str(3)], axis=0)

    return params['X'], params['Y']

def import_test_data():
    params = {}

    params['X'] = unpickle('data/test_batch')[b'data']
    params['Y'] = unpickle('data/test_batch')[b'labels']
    params['Y'] = np.append(params['Y'], [], axis=0)

    return params['X'], params['Y']

#(X_train, Y_train) = import_training_data()
#(X_test, Y_test) = import_test_data()

#np.savetxt('X_train.csv', X_train, delimiter=',')
#np.savetxt('Y_train.csv', Y_train, delimiter=',')
#np.savetxt('X_test.csv', X_test, delimiter=',')
#np.savetxt('Y_test.csv', Y_test, delimiter=',')