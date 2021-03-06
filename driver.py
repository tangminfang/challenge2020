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

# num1=0
# num2=0
# num3=0
# num4=0
# num5=0
# num6=0
# num7=0
# num8=0
# num9=0
def getdata_class(header):
    # global num1,num2,num3,num4,num5,num6,num7,num8,num9
    classes=set()
    label=np.zeros([1,9],dtype='int')
    for lines in header:
        if lines.startswith('#Dx'):
            tmp = lines.split(': ')[1].split(',')
            for c in tmp:
                classes.add(c.strip())
            classname=list(classes)
            classname0=classname[0]
            # classnamemul=0
            # if len(classname)>1:
            #     classnamemul=classname
            i=0
            while i<len(classname):
                if(classname[i]=="Normal"):
                    labelclass = 4
                    # num1=num1+1
                elif(classname[i]=="AF"):
                    labelclass = 1
                    # num2=num2+1
                elif (classname[i] == "I-AVB"):
                    labelclass = 2
                    # num3=num3+1
                elif (classname[i] == "LBBB"):
                    labelclass = 3
                    # num4=num4+1
                elif (classname[i] == "RBBB"):
                    labelclass = 7
                    # num5=num5+1
                elif (classname[i] == "PAC"):
                    labelclass = 5
                    # num6=num6+1
                elif (classname[i] == "PVC"):
                    labelclass = 6
                    # num7=num7+1
                elif (classname[i] == "STD"):
                    labelclass = 8
                    # num8=num8+1
                elif (classname[i] == "STE"):
                    labelclass = 9
                    # num9=num9+1

                i=i+1
                labelclass=labelclass-1
                label[0,labelclass]=1

            if (classname0 == "Normal"):
                label0 = 4
            elif (classname0 == "AF"):
                label0= 1
            elif (classname0 == "I-AVB"):
                label0 = 2
            elif (classname0 == "LBBB"):
                label0 = 3
            elif (classname0 == "RBBB"):
                label0 = 7
            elif (classname0 == "PAC"):
                label0 = 5
            elif (classname0 == "PVC"):
                label0 = 6
            elif (classname0 == "STD"):
                label0 = 8
            elif (classname0 == "STE"):
                label0 = 9
            label0=label0-1
    # print(num1,num2,num3,num4,num5,num6,num7,num8,num9)
    return label,label0

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
