

# 2. Tutorial Video 

## 2.1 https://www.youtube.com/watch?v=_L39rN6gz7Y Decision Tree and Classification

- When a Decision Tree classifies things into categories, it is called the Classification Tree
- When a Decision Tree predicts numeric values, it is called the Regression Tree
- The very top of the tree is called 'Root Node' or just 'Root'
- The very end of the tree is called 'Leaf Nodes' or just 'Leaves'
- The nodes between Root and Leaves are called 'Internal Nodes' or 'Branches'

The leaf that contains a mixture of labels is called 'impure leaf'
The leaf that contains only one label is called 'pure leaf'

There are several ways to quantify the impurity of leaves:
1. Gini Impurity
2. Entropy
3. Information Gain

Gini Impurity for Categorical Feature:
1. Gini Impurity for single leaf = 1 - square of the probability of positive - square of the probability of negative
2. Gini Impurity for the Categorical feature = weighted average of Gini Impurities for the leaves

Gini Impurity for Numerical Feature:
1. sort the rows of this numerical feature from lowest to highest
2. calculate the average value for all adjacent values
3. calculate the Gini impurity for every average value

Compare the Gini Impurity among both numerical features and categorical features, and pick the one with lowest Gini Impurity as the root, then, continue the Gini Impurity calculation to determine the brances until the final leaf

Just note, the output of leaf is the label with most votes

To avoid the overfitting in decision tree, we can:
1. Pruning
2. We can put limits on how trees grow, for example, by requiring 3 or more rows per leaf

Also Note: Whe we build a tree, we don't know in advance if it is better to require 3 data rows per leaf or some other number, so, we test different values with something called 'Cross Validation' and pick the one that works best.

## 2.2 https://www.youtube.com/watch?v=wpNl-JwwplA Feature Selection and Missing Data

How to deal with missing data:
1. Just delete
2. If it is categorial feature, we can just impute this missing value with the label with most votes
3. If it is numerical feature, we can either use mean value or median value to impute this missing data
4. If there is high correlation between this numerical feature and another numerical feature, we can fit linear regression

The feature selection in decision tree is based on the comparison of Gini Imuprity, the lowest the better

## 2.3 https://www.youtube.com/watch?v=g9c66TUylZ4 Regression Tree

- When we need to use something other than a straight line to make predictions, one option is to use the Regression Tree
- In a Regression Tree, each leaf represents a numeric value which is calculated from the average value of classified data rows from training dataset
- To find out the best threshold as the root for numeric predictor, firstly, we need to sort the predictor from lowest to highest, and calculate the adjacent average values, then, for each average value, we use it as root and build a simple tree, then, we need to calculate the sum of square of residuals, finally, we will use the threshold with lowest sum of square residuals as the root
- Overall, in regression tree, we build the tree by comparing the sum of square residuals

## 2.4 https://www.youtube.com/watch?v=D0efHEJsfHo How to Prune Regression Trees

Cost Complexity Pruning

Calculate the Sum of Squared Residuals for each tree or sub-tree

Weakest Link Pruning works by calculating a 'Tree Score' which is based on the Sum of Squared Residuals (SSR) for the tree or sub-tree and a 'Tree Complexity Penalty' that is a function of the number of leaves in the tree or sub-tree

Test Score = SSR + alpha*T (Tree Complexity Penalty)

Note: alpha is a tuning parameter that we finding using Cross Validation

## 2.5 https://www.youtube.com/watch?v=J4Wdy0Wc_xQ Random Forest

- Random Forest is made out of decision trees
- Decision Trees are easy to build, easy to use and easy to interpret, but in practice they are not that awesome
- In other works, Decision Trees work great with the data used to create them, but they are not flexible when it comes to classfying new samples
- The good news is that Random Forests combine the simplicity of decision trees with flexibility resulting in a vast improvement in accuracy

Step 1: Create a 'Bootstrapped' Dataset
- To create a bootstrapped dataset that is the same size as the original, we just randomly select samples from the original dataset
- The important detail is that we're allowed to pick the same sample more than once

Step 2: Create a decision tree using the bootstrapped dataset, but only use a random subset of variables (or columns) at each step. For example, you ramdonly select 2 variables to determine which variable is best for root, then, for the remaining variables, you randomly select 2 variables to determine which variable is best for branch. Here the number '2' is a tuning parameter. 

Using a bootstrapped sample and considering only a subset of the variables at each step results in a wide variety of trees. The variety is what makes random forests more effective than individual decision trees

Note: Bootstrapping the data plus using the aggregate to make a decision is called 'Bagging'

![image](https://user-images.githubusercontent.com/60442877/149911253-0495c15f-1ee2-49ec-8bea-2134b20701a0.png)

## 2.6  Random Forest: Missing Data and Clustering

https://www.youtube.com/watch?v=sQ870aTKqiM

- For the missing data, we firstly have an initial guess for those missing values, then, we need to refine our guess
- We refine the guess by determining which samples are similar to the one with missing data, so we need to determine the similarity first
- Step 1: Build a random forest
- Step 2: Run all of the data down all of the trees and we keep track of similar samples using a 'Proximity Matrix'
- Step 3: We use the proximity values to make better guessses about the missing data
- After we revised our guesses, we do the whole steps again, we do this 6 or 7 times until the missing values converge (i.e. no longer change each time we recalculate)

1 - the proximity values = distance between samples which result in 'Distance Matrix'

## 2.7 AdaBoost

https://www.youtube.com/watch?v=LsK-xG1cLYA

- In a Random Forest, each time you make a tree, you make a full sized tree
- In contrast, in a 'Forest of Trees' made with AdaBoost, the treesd are usually just a node and two leaves (Stump)
- So, 'Forest of Trees' is actually Forest of Stumps
- Stumps are not great at making accurate classifications since it can only use one variable to make a decision. Thus, Stumps are technically weak learners
- In a Random Forest, each tree has an equal vote on the final classification
- In contrast, in a Forest of Stumps made with AdaBoost, some stumps get more say in the final classification than others.
- In a Random Forest, each decision tree is made independently of the others, in contrast, in a Forest of Stumps made with AdaBoost, order is important

To review, the three ideas behind AdaBoost are:
1. AdaBoost combines a lot of 'Weak Learners' to make classifications. The weak learners are almost always stumps
2. Some stumps get more say in the classification than others
3. Each stump is made by taking the previous stump's mistakes into account

Amount of Say = (1/2)*log[(1-totalerror)/totalerror]

TotalError = the sum of sample weights of error rows

New Sample Weight of misclassified samples = sample weight * exp(amount of say)

New Sample Weight of correctly classified samples = sample weight * exp(-amount of say)

We also need to normalize the New Sample Weight so that they add up to 1

In theory, we could use the Sample Weights to calculate 'Weighted Gini Indexes' to determine which variable should split the next stump

Alternatively, instead of using a Weighted Gini Index, we can make a new collection of samples that contains duplicate copies of the samples by using the sample weights, and the new collection of samples should be same size as the original, and we just give all the samples equal sample weights and build stump like before

For AdaBoost, the final classification result is determined by comparing the total amount of say between the predicted positive and negative labels

## 2.8 Gradient Boost for Regression 

https://www.youtube.com/watch?v=3CC4N4z3GJc

- When Gradient Boost is used to predict a continuous value, we say that we are using Gradient Boost for Regression
- Using Gradient Boost for regression is different from doing linear regression
- Gradient Boost starts by making a single leaf instead of a tree or stump, this leaf represents an initial guess of all the samples
- Like AdaBoost, Gradient Boost builds fixed sized trees based on the previous tree's error, but unlike AdaBoost, each tree can be larger than a stump
- Gradient Boost builds another tree based on the errors made by previous tree, and Gradient Boost continuous to build trees in this fashion until it has made the number of trees you asked for, or additional trees fail to improve the fit


1. Step 1: Build a single leaf which is generated by calculating the average value of whole samples
2. Step 2: Build a tree based on the errors from the first tree, the errors are called 'Pseudo Residuals'. And we want to build a tree to predict these residual
3. Step 3: We can combine the orignial leaf with new tree to make a new prediction, thats the average value + learning rate * predicted residual value
4. Step 4: Build another tree based on the errors made by previous tree, and do prediction again, the prediction value = the average value + learning rate * predicted residual value from first tree + learning rate * predicted residual value from second tree
5. Step 5: Then, we repeart above steps to make trees until we reach the maximum specified or adding additional trees does not significantly reduce the size of the residuals

Gradient Boost deals with overfitting problem by using a Learning Rate to scale the contribution from the new tree that is used to predict the residuals, and the learning rate is a value between 0 and 1

The final predicted value = average value + learning rate * predicted Residual

https://www.youtube.com/watch?v=2xudPOBz-vs

- Loss Function is just something that evaluates the prediction performance
- The Loss Function for the regression in Gradient Boost is: (1/2)*square of the difference between observed and predicted values
- Gradient Boost usually uses trees larger than stumps

## 2.9 Gradient Boost for Classification

### 2.91 Overview 

https://www.youtube.com/watch?v=jxuNLH5dXCs

- We start with a leaf that represents an initial log(odds) prediction which is just the log of the number of positive labels being numerator and the number of negative labels being denominator, with logistic function, we can also get the initial probability prediction 
- We can measure how bad the initial prediction is by calculating Pseudo Residuals, the difference between the observed and the predicted probability values
- Then, we will build a Tree to predict residuals, and we use the following formula to transform these leaf values into the outputs which will be used to predict new log(odds)
![image](https://user-images.githubusercontent.com/60442877/150294561-b5c77ca7-b588-492d-92fc-e87a96fe618c.png)
- Then, the output of each leaf will be used to predict new log(odds)
- New log(odds) prediction = previous log(odds) + learning rate * output of decision tree
- Then, we input these new log(odds) into the logistic function to get new probability prediction
- Finally, we calculate the Pseudo Residuals as before, build the tree again, and repeat until we have made the maximum number of trees specified, or the residuals get super small


NOTE: 
1. In Gradient Boost, we limit the number of leaves in the trees we built
2. In Gradient Boost for classification, we predict the log(odds), so the predicted value need to be transformed into probability by using logistic function
3. Gradient Boost usually uses trees with between 8 to 32 leaves
4. In practice, the tree we build is 100 and more

### 2.92 Math Details

https://www.youtube.com/watch?v=StWY5QWMXCw

The log-likelihood of the observed data given the predicted probability is:

![image](https://user-images.githubusercontent.com/60442877/150310972-2f03af24-bf31-4fb8-a7d1-15f55f0e7785.png)

If we want to use log-likelihood as a Loss Function, where smaller values represent better fitting models, then, we need to multiply the log-likelihood by -1

## 2.10 XGBoost for Regression

https://www.youtube.com/watch?v=OtD8wVaFm6E

- XGBoost is desgined to be used with large, complicated datasets
- The very first step in fitting XGBoost to the training dataset is to make an initial prediction
- This initial prediction can be anything, but by default, it is 0.5, regardless of whether you are using XGBoost for Regression or Classification
- We start out the XGBoost tree from a single leaf and all the residuals (observed - initial prediction) go to that leaf
- Then, we will calculate 'Quality Score' or 'Similarity Score' for the Residuals
![image](https://user-images.githubusercontent.com/60442877/150336402-838fed4a-937d-4f79-8447-2d45dfc1bfda.png)
where the lambda is a regularization parameter
- Then, we need to calculate the Gain values of splitting the Residuals into two groups by different thresholds, better splitting has higher gain value
![image](https://user-images.githubusercontent.com/60442877/150337454-d93464e5-2e30-4639-8da7-b530c5023270.png)
- Then, we will choose the threshold with the highest gain value for the root in the tree
- After that, we will find out the branches by comparing the gain values
- Just notice, by default setting, we limit the tree depth to 6 levels
- Then, we will prune the XGBoost tree based on the Gain values, we start by picking a number denoted as 'gamma' which is called 'Tree Complexity Parameter', we then calculate the difference between the Gain and the gamma, if the difference is positive, keep the node, otherwise, remove node. We start the calculattion from the lowest bracch in the tree. If the difference in root is negative, but the branch is positive, keep the root
- Remember, lambda is a Regularization Parameter, which means that it is intended to reduce the prediction's sensitivity to individual observations, and when lambda > 0, it is easier to prune leaves because the values for Gain are smaller
- Lambda in the formula of similarity score help us avoid overfitting the data
- After the XGBoost tree built, we need to determine the output value for each leaf, which is given as following:
![image](https://user-images.githubusercontent.com/60442877/150340760-62aa307a-4788-451d-b0b9-4538b4b820be.png)
- Just like Gradient Boost, XGBoost makes new predictions by starting with the initial prediction and adding the output of the tree scaled by a learning rate
- In XGBoost, the learning rate is called 'eta', and the default value is 0.3
- Then, we can get new prediction for each observation, and obtain new residuals, and we will use these residuals to build new XGBoost trees until the Residuals are super small or we have reached the maximum number

## 2.11 XGBoost for Classification 

https://www.youtube.com/watch?v=8b1JEDvenQU

- The very first step in fitting XGBoost to the Training dataset is to make an initial prediction
- This prediction can be anything, for example, it can be the sample proportion of being positive in traning data, but by default, it is just 0.5, regardless of whether you are using XGBoost for Regression or Classification
- Then, we calculate the Residual which is the observed probability - initial probability
- Then, we fit an XGBoost tree to these residuals
- Similar to the XGBoost for regression, we also need the similarity score to build the tree, but different from the regression XGBoost, the similarity score for classification is different formua which is given below:
![image](https://user-images.githubusercontent.com/60442877/150359920-9adbc80e-4195-4635-b965-eccd2620207c.png)
- Then, just like what we did in XGBoost for regression, we calculate the similarity scores and gain values for each threshold, and build the XGBoost Tree
- Then, we will do pruning by calculating the difference between gain values and gamma (tree complexity parameter)
- The output of leaf of XGBoost tree for classification is given below which is just the sum of residuals over the sum of the previously predicted probability*(1 - previously predicted probability) + lambda
![image](https://user-images.githubusercontent.com/60442877/150364563-0ee58ad8-2dca-4c78-aa8c-55df4748ef3c.png)
- Finally, we will make predictions with initial log(odds)+ learning rate* the output from tree, and put the predicted value into the logistic function to get the predicted probability
- Then, we calculate the new residual, and build XGBoost tree again until we reach the maximum number of the tree or the residuals are very small


Big Notice: XGBoost also has a threshhold for the minimum number of Residuals in each leaf which is determined by calculating one metric called 'Cover' (in python, that is the parameter min_child_weight), and the default value is 1. In XGBoost for regression, the Cover is just equal to the number of resisuals in the leaf. In XGBoost for classification, the Cover is equal to the sum of the previously predicted probability*(1 - previously predicted probability) for each residual in the leaf. If the cover in one leaf is less than our specified value, the leaf will be removed.


## 2.12 XGBoost Math Detail

https://www.youtube.com/watch?v=ZVFeW798-2I

Big Notice: 
1. For regression, the loss function is just the square of residual for each observation
2. For classification, the loss function is -{yi*log(pi) + (1-yi)*log(1-pi)} for binary classification, actually, the loss function is just the predicted probability being the true label

## 2.13 Crazy Cool Optimizations about XGBoost

https://www.youtube.com/watch?v=oRrKeUCEbq8

### Approximate Greedy Algorithm

- By using a Greedy Algorithm, XGBoost can build a tree relatively quickly. That means, when we have a lot of observations or feature variables, then, the Greedy Algorithm becomes slow or even take forever because it still has to look at every possible threshold
- This is where the 'Approximate Greedy Algorithm' comes in
- When we have a lot of observations for one single numerical feature, instead of testing every single threshold, we could divide the data into Quantiles and only use quantiles as candidate thresholds to split the observations.
- However, the more quantiles we have, the more thresholds we will have to test, and that means it will take longer to build the tree
- For XGBoost, the 'Approximate Greedy Algorithm' means that instead of testing all possible thresholds, we only test the quantiles
- Bt default, the approximate greedy algrotihm uses about 33 quantiles

### Parallel Learning and Weighted Quantile Sketch

- When you have tons and tons of data, so much data that you can't fit it all into a computer's memory at one time, then things that looks simple, like sorting a list of numbers and finding quantiles, become really slow
- To get around this problem, a class of algorithms, called 'Sketches', can quickly create approximate solutions
- For example, imagine we are just using a ton of Dosages to predict Drug Effectiveness, and imagine splitting this big dataset into small pieces and putting the pieces on different computers on a network
- The 'Quantile Sketch Algorithm' combines the values from each computer to make an approximate histogram, then the approximate histogram is used to calculate approximate quantiles, and the approximate greedy algorithm uses approximate quantiles
- With weighted quantiles, each observation has a corresponding weight and the sum of the weights are the same in each quantile
- When using XGBoost for classification, the weights for the Weighted Quantile Sketch are calculated from the previously predicted probabilities*(1-previously predicted probabilities)
- When using XGBoost for regression, the weights for the Weighted Quantile Sketch is just the number of residuals per observation which is just 1
- So, instead of using equal quantiles, XGBoost tries to make quantiles that have a similar sum of weights

Overall, when we have a huge training dataset, XGBoost uses an Approximate Greedy Algorithm and that means using Parallel Learning to split up the dataset so that multiple computers can work on it at the same time, and a Weighted Quantile Sketch merges the data into an approximate histogram, and the histogram is divided into weighted quantiles that put observations with low confidence predictions into quantiles with fewer observations

Note: XGBoost only uses the Approximate Greedy Algrorithm, Parallel Learning and the Weighted Quantile Sketch when the training dataset is huge. When the Training datasets are small, XGBoost just uses greedy algorithm

### Sparsity-Aware Split Finding

- Sparsity-Aware Split Finding tells us how to build trees with missing data and how to deal with new observations when there is missing data

### Cache-Aware Access

- The basic idea is that inside each computer we have a CPU and that CPU has a small amount of Cache Memory. The CPU can use this memory faster than any other memory in the computer. The CPU is also attached to a large amount of Main Memory. While the main memory is larger than the cache, it takes longer to use. Lastly, the CPU is also attached to the Hard Drive. The Hard Drive can store the most stuff, but is the slowest of all memory options. 
- If you want your program to run really fast, the goal is to maximize what you can do with the cache memory
- So, XGBoost puts the Gradients and Hessians in the Cache so that it can rapidly calculate Similarity Score and Output Values.


### Blocks for Out-of-Core Computation

- When the dataset is too large for the Cache and Main Memory, then, at least some of it, must be stored on the Hard Drive
- Because reading and writing data to the Hard Drive is super slow, XGBoost tries to minimize these actions by compressing the data
- Even though the CPU must spend some time decompressing the data that comes from the Hard Drive, it can do this faster than the Hard Drive can read the data
- In other words, by spending a little bit of CPU time uncompressing the data, we can avoid spending a lot of time accessing the Hard Drive
- Also, when there is more than one Hard Drive available for storage, XGBoost uses a database technique called Sharding to speed up disk access
