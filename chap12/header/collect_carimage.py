import numpy as np
import cv2

def SVM_create(type, max_iter, epsilon):
    svm = cv2.ml.SVM_create()
    ## SVM IRDE T
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setGamma(1)
    svm.setC(1)
    svm.setTermCriteria((type, max_iter, epsilon))
    return svm

nsample = 140
trainData = [cv2.imread("../images/plate/%03d.png" %i, 0) for i in range(nsample)]
trainData = np.reshape(trainData, (nsample, -1)).astype('float32')
labels = np.zeros((nsample, 1), np.int32)
labels[:70] = 1


print("SVM 객체 생성")
svm = SVM_create(cv2.TERM_CRITERIA_MAX_ITER, 1000, 1-6)
svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)
svm.save("SVMtrain.xml")
print("SVM 객체 저장 완료")