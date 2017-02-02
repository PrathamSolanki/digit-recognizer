import graphlab

#Loading Data files
train_data = graphlab.SFrame('../../../Data/train.csv')
test_data = graphlab.SFrame('../../../Data/test.csv')

#Splitting Training data into Training and Cross Validation sets
train, cross_validation = train_data.random_split(0.8)

#Training Boosted Trees Classifier ont the Training Set
digitRecognizer_BT = graphlab.boosted_trees_classifier.create(train, target='label', max_iterations=100, validation_set=None)

#Evaluating Model on the Cross Validation set
validation_accuracy = digitRecognizer_BT.evaluate(cross_validation)

print "Validation set accuracy: ", validation_accuracy['accuracy']

#Making Predictions on the Test set
pred = digitRecognizer_BT.predict(test_data)

#Writing predictions to file
pred.save('submission.csv', format='csv')
