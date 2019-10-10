# NLP_comment_cluster
the goal is to divide the comment into different situations and use the result to analyze deeply.
comment text need to be devide into some class, it seems like a classfication, but in fact, it's a clustering problem.

in this solution, i divide two steps to do:

first, use tf-idf method to extract the comment text features;

second, use the simple k-means method to cluster the feature matrix from first step;
