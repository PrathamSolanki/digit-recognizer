echo "\nLoading data and Pre-Processing it into the LIBSVM data format....\n"
python csv2libsvm_python2.py ../../../Data/train.csv train.data 0 True
python csv2libsvm_python2.py ../../../Data/test.csv test.data -1 True

echo "Traing Support Vector Machine....\n"
./svm-train -s 0 -t 1 train.data digitRecognizer.model

echo "Making Predictions on Test data....\n"
./svm-predict test.data digitRecognizer.model predictions.csv

echo "Writing Predictions to file (submission.csv)....\n"
octave readyingSubmission.m