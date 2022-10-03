from sklearn.metrics import plot_confusion_matrix
import warnings
import torch
#import tqdm
#import sys
import torch
from sklearn.metrics import classification_report
import config
#import time
from model2 import TUMORCLASSIFIER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataset_loader import data_loader
import pdb
#device='cpu'

def accuracy(outputs, labels):
    """
    Args:
        outputs: predictions from the model
        labels: original labels
    return:
        correct: total number of correct classifications
    """
    outputs = outputs
    labels = labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        predictions = torch.max(outputs, 1)[1]
        correct = (predictions == labels).sum()

        acc = correct / len(predictions)
        acc = acc.cpu().numpy()
        return acc


    
def evaluate_proposed_model():
    """
    Args:
        model: trained model
        test_loader: test dataset loader
        mode: binary or multiclass
    """
    
    a, b, test_loader = data_loader()
    
    del a, b
    #pdb.set_trace()
    model = TUMORCLASSIFIER()
    model = model.to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    model.eval()
    
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        count = 1
        with torch.no_grad():
            correct = 0
            total_samples = 0

            #test_bar = tqdm(test_loader, file=sys.stdout)

            for samples, labels in test_loader:

                samples, labels = samples.to(device), labels.to(device)
                #pdb.set_trace()
                output = model(samples)

                print("Report on BATCH: ", count)
                
                predictions = torch.max(outputs, 1)[1]
    
                detailed_report = classification_report(predictions.cpu().numpy(), labels.cpu().numpy())
                print(detailed_report)
                
                #classification_report_multi(output, labels)
                #report(output, labels)
                count += 1

                #start_time = time.time()
                #output = model(samples)
                #execution_time = time.time() - start_time
                print('*****************************************')
                #print("Execution time: {}, Per sample: {}".format(execution_time, execution_time / len(samples)))


                
def report(outputs, labels):
    
    predictions = torch.max(outputs, 1)[1]
    
    detailed_report = classification_report(predictions.cpu().numpy(), labels.cpu().numpy())
    print(detailed_report)
    
    
#evaluate_proposed_model()