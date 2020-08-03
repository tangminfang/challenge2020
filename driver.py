#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from get_12ECG_features import get_12ECG_features

def load_challenge_data(filename):


    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

  
# Find unique number of classes  
def get_classes(input_directory,files):

    classes=set()
    for f in files:
        g = f.replace('.mat','.hea')
        input_file = os.path.join(input_directory,g)
        with open(input_file,'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)

def getdata_class(header):

    classes=set()
    for lines in header:
        if lines.startswith('#Dx'):
            tmp = lines.split(': ')[1].split(',')
            for c in tmp:
                classes.add(c.strip())
            classname=list(classes)[0]
            if(classname=="Normal"):
                label=4
            elif(classname=="AF"):
                label=1
            elif (classname == "I-AVB"):
                label=2
            elif (classname == "LBBB"):
                label=3
            elif (classname == "RBBB"):
                label=7
            elif (classname == "PAC"):
                label=5
            elif (classname == "PVC"):
                label=6
            elif (classname == "STD"):
                label=8
            elif (classname == "STE"):
                label=9

    return label

# if __name__ == '__main__':
def dataread():
    # Parse arguments.
    # if len(sys.argv) != 3:
    #     raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    # input_directory = sys.argv[1]
    input_directory = '/usr/games/Training_WFDB'
    # output_directory = sys.argv[2]

    # Find files.
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    # if not os.path.isdir(output_directory):
    #     os.mkdir(output_directory)

    classes=get_classes(input_directory,input_files)

    # # Load model.
    # print('Loading 12ECG model...')
    # model = load_12ECG_model()

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)
    datalabel=[]
    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory,f)
        data,header_data = load_challenge_data(tmp_input_file)
        datalabel.append(getdata_class(header_data))
        features = np.asarray(get_12ECG_features(data, header_data))
        feats_reshape = features.reshape(1, -1)
        # np.save('/datanpy/A'+'%04d.npy'%(i+1), data)
        # current_label, current_score = run_12ECG_classifier(data,header_data,classes, model)
        # # Save results.
        # save_challenge_predictions(output_directory,f,current_score,current_label,classes)

    datalabel=np.array(datalabel)
    # np.save('/usr/games/datanpylabel/recordlabel.npy', datalabel)
    print('Done.')
    return data,datalabel
