library(plyr)

X = read.csv("../imageFeatures/outputFeatures/image_features.csv")
labels = read.csv("../data/labels/label.csv")
hist(labels$X0)
