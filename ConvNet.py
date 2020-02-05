import os
import cv2
import numpy as np
import tqdm as tqdm

rebuild_data = False

class DogsVsCats():
    # can this have an effect also?
    def __init__ (self):
        self.IMG_SIZE = 50
        #read in ims
        self.CATS = "PetImages\\Cat"
        self. DOGS = "PetImages\\Dog"
        self.Labels = {
            self.CATS: 0,
            self.DOGS: 1
        }
        self.training_data = []
        self.catCount = 0
        self.dogCount = 0
        self.cwd = os.getcwd()

    def makeTrainingData(self):
        for label in self.Labels:
            print(label)
            for f in tqdm.tqdm(os.listdir(label)):
                try:
                    path = os.path.join(self.cwd, label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    print(img)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    '''
                    np.eye makes one hot vectors easy where you can call using the index
                    eg. np.eye(2)[1] = [1,0]. np.eye(2)[1,0]
                    '''
                    #getting the info from dict
                    self.training_data.append([np.array(img), np.eye(2)[self.Labels[label]]])

                    if label == self.CATS:
                        self.catCount += 1
                    elif label == self.DOGS:
                        self.dogCount += 1

                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("trainingdata.npy", self.training_data)
        print("CATS:", self.catCount)
        print("DOGS:", self.dogCount)

if rebuild_data:
    dogsvcats=DogsVsCats()
    dogsvcats.makeTrainingData()

training_data=np.load("trainingdata.npy", allow_pickle=True)
print(len(training_data))
print(training_data[0])
