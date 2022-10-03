import os
import pdb

DATA_FOLDER = os.getcwd()+'/Data/'

TRAIN_DATA_LABELS = DATA_FOLDER + 'labels_train.csv'
TEST_DATA_LABELS = DATA_FOLDER + 'labels_test.csv'

TRAIN_DATA_FOLDER = DATA_FOLDER + '/train'
TEST_DATA_FOLDER = DATA_FOLDER + '/test'


NUMBER_OF_CLASSES = 33

#pdb.set_trace()

BEST_MODEL_PATH = os.getcwd() + '/PretrainedModel/model2.pth'

EPOCHS = 1000
BATCH_SIZE = 99
LEARNINIG_RATE = 0.001

TRAINING = True
TESTING = False

