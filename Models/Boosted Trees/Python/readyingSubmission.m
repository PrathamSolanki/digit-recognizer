pred = csvread('submission.csv');
ImageId = 1:1:size(pred);
output = [ImageId' pred];
csvwrite ('submission.csv', output);
