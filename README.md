# 1. Decision Tree

## 1.1 Introduction

Classification is a two-step process, learning step and prediction step, in machine learning. In the learning step, the model is developed based on given training data. In the prediction step, the model is used to predict the response for given data. Decision Tree is one of the easiest and popular classification algorithms to understand and interpret.

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).

In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

A decision tree is drawn upside down with its root at the top. In the image on the left, the bold text in black represents a condition/internal node, based on which the tree splits into branches/ edges. The end of the branch that doesn’t split anymore is the decision/leaf, in this case, whether the passenger died or survived, represented as red and green text respectively.

![image](https://user-images.githubusercontent.com/60442877/147900468-0addb3d1-35a1-465e-ab14-f505ccf4ee12.png)

Although, a real dataset will have a lot more features and this will just be a branch in a much bigger tree, but you can’t ignore the simplicity of this algorithm. The feature importance is clear and relations can be viewed easily. This methodology is more commonly known as learning decision tree from data and above tree is called Classification tree as the target is to classify passenger as survived or died. Regression trees are represented in the same manner, just they predict continuous values like price of a house. 

In general, Decision Tree algorithms are referred to as CART or Classification and Regression Trees.

So, what is actually going on in the background? Growing a tree involves deciding on which features to choose and what conditions to use for splitting, along with knowing when to stop. As a tree generally grows arbitrarily, you will need to trim it down for it to look beautiful. Lets start with a common technique used for splitting.

## Types of Decision Trees

Types of decision trees are based on the type of target variable we have. It can be of two types:

- Categorical Variable Decision Tree: Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.
- Continuous Variable Decision Tree: Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree.

## 1.2 Important Terminology related to Decision Trees

- Root Node: It represents the entire population or sample and this further gets divided into two or more homogeneous sets.
- Splitting: It is a process of dividing a node into two or more sub-nodes.
- Decision Node: When a sub-node splits into further sub-nodes, then it is called the decision node.
- Leaf / Terminal Node: Nodes do not split is called Leaf or Terminal node.
- Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say the opposite process of splitting.
- Branch / Sub-Tree: A subsection of the entire tree is called branch or sub-tree.
- Parent and Child Node: A node, which is divided into sub-nodes is called a parent node of sub-nodes whereas sub-nodes are the child of a parent node.

![image](https://user-images.githubusercontent.com/60442877/147969331-de7dfcbd-3257-4fba-8a74-f907232c5152.png)

Decision trees classify the examples by sorting them down the tree from the root to some leaf/terminal node, with the leaf/terminal node providing the classification of the example.

Each node in the tree acts as a test case for some attribute, and each edge descending from the node corresponds to the possible answers to the test case. This process is recursive in nature and is repeated for every subtree rooted at the new node.

The primary challenge in the decision tree implementation is to identify which attributes do we need to consider as the root node and each level. Handling this is to know as the attributes selection. We have different attributes selection measures to identify the attribute which can be considered as the root note at each level.

## 1.3 How do Decision Trees work?

The decision of making strategic splits heavily affects a tree’s accuracy. The decision criteria are different for classification and regression trees.

Decision trees use multiple algorithms to decide to split a node into two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that the purity of the node increases with respect to the target variable. The decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.

The algorithm selection is also based on the type of target variables. Let us look at some algorithms used in Decision Trees:

![image](https://user-images.githubusercontent.com/60442877/147969965-7c082d49-1428-43f9-aa11-4f3d917f598d.png)

The ID3 algorithm builds decision trees using a top-down greedy search approach through the space of possible branches with no backtracking. A greedy algorithm, as the name suggests, always makes the choice that seems to be the best at that moment.


## 1.4 Attribute Selection Measures

If the dataset consists of N attributes then deciding which attribute to place at the root or at different levels of the tree as internal nodes is a complicated step. By just randomly selecting any node to be the root can’t solve the issue. If we follow a random approach, it may give us bad results with low accuracy.

For solving this attribute selection problem, researchers worked and devised some solutions. They suggested using some criteria like :

- Entropy：a measure of the randomness in the information being processed. The higher the entropy, the harder it is to draw any conclusions from that information.
- Information gain： a statistical property that measures how well a given attribute separates the training examples according to their target classification.
- Gini index： You can understand the Gini index as a cost function used to evaluate splits in the dataset. 
- Gain Ratio
- Reduction in Variance： an algorithm used for continuous target variables (regression problems). This algorithm uses the standard formula of variance to choose the best split. The split with lower variance is selected as the criteria to split the population.
- Chi-Square

These criteria will calculate values for every attribute. The values are sorted, and attributes are placed in the tree by following the order i.e, the attribute with a high value(in case of information gain) is placed at the root.

While using Information Gain as a criterion, we assume attributes to be categorical, and for the Gini index, attributes are assumed to be continuous.

## 1.5 Recursive Binary Splitting

In this procedure all the features are considered and different split points are tried and tested using a cost function. The split with the best cost (or lowest cost) is selected.

Consider the earlier example of tree learned from titanic dataset. In the first split or the root, all attributes/features are considered and the training data is divided into groups based on this split. We have 3 features, so will have 3 candidate splits. Now we will calculate how much accuracy each split will cost us, using a function. The split that costs least is chosen, which in our example is sex of the passenger. This algorithm is recursive in nature as the groups formed can be sub-divided using same strategy. Due to this procedure, this algorithm is also known as the greedy algorithm, as we have an excessive desire of lowering the cost. This makes the root node as best predictor/classifier.

## 1.6 Cost of a split

Lets take a closer look at cost functions used for classification and regression. In both cases the cost functions try to find most homogeneous branches, or branches having groups with similar responses. This makes sense we can be more sure that a test data input will follow a certain path.

![image](https://user-images.githubusercontent.com/60442877/147963728-14371960-5fce-4f65-9121-783590127833.png)

Lets say, we are predicting the price of houses. Now the decision tree will start splitting by considering each feature in training data. The mean of responses of the training data inputs of particular group is considered as prediction for that group. The above function is applied to all data points and cost is calculated for all candidate splits. Again the split with lowest cost is chosen. 

## 1.7 Gini Score

A Gini score gives an idea of how good a split is by how mixed the response classes are in the groups created by the split. Here, pk is proportion of same class inputs present in a particular group. A perfect class purity occurs when a group contains all inputs from the same class, in which case pk is either 1 or 0 and G = 0, where as a node having a 50–50 split of classes in a group has the worst purity, so for a binary classification it will have pk = 0.5 and G = 0.5.

![image](https://user-images.githubusercontent.com/60442877/147964100-87e5f82f-bf6a-4c63-9baa-4299c9c5f495.png)

## 1.8 When to stop splitting?

You might ask when to stop growing a tree? As a problem usually has a large set of features, it results in large number of split, which in turn gives a huge tree. Such trees are complex and can lead to overfitting. So, we need to know when to stop? One way of doing this is to set a minimum number of training inputs to use on each leaf. For example we can use a minimum of 10 passengers to reach a decision(died or survived), and ignore any leaf that takes less than 10 passengers. Another way is to set maximum depth of your model. Maximum depth refers to the the length of the longest path from a root to a leaf.


## 1.9 Advantage

- Simple to understand and to interpret. Trees can be visualised.
- Requires little data preparation. Other techniques often require data normalisation, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
- The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
- Able to handle both numerical and categorical data. However scikit-learn implementation does not support categorical variables for now. Other techniques are usually specialised in analysing datasets that have only one type of variable. See algorithms for more information.
- Able to handle multi-output problems.
- Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
- Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
- Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

## 1.10 Disadvantage

- Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
- Predictions of decision trees are neither smooth nor continuous, but piecewise constant approximations. Therefore, they are not good at extrapolation.
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.


## 1.11 How to avoid overfitting in Decision Trees?

The common problem with Decision trees, especially having a table full of columns, they fit a lot. Sometimes it looks like the tree memorized the training data set. If there is no limit set on a decision tree, it will give you 100% accuracy on the training data set because in the worse case it will end up making 1 leaf for each observation. Thus this affects the accuracy when predicting samples that are not part of the training set.

Here are two ways to remove overfitting:

- Pruning Decision Trees.
- Random Forest

### 1.111 Pruning

The performance of a tree can be further increased by pruning. It involves removing the branches that make use of features having low importance. This way, we reduce the complexity of tree, and thus increasing its predictive power by reducing overfitting.

Pruning can start at either root or the leaves. The simplest method of pruning starts at leaves and removes each node with most popular class in that leaf, this change is kept if it doesn't deteriorate accuracy. Its also called reduced error pruning. More sophisticated pruning methods can be used such as cost complexity pruning where a learning parameter (alpha) is used to weigh whether nodes can be removed based on the size of the sub-tree. This is also known as weakest link pruning.

In pruning, you trim off the branches of the tree, i.e., remove the decision nodes starting from the leaf node such that the overall accuracy is not disturbed. This is done by segregating the actual training set into two sets: training data set, D and validation data set, V. Prepare the decision tree using the segregated training data set, D. Then continue trimming the tree accordingly to optimize the accuracy of the validation data set, V.

![image](https://user-images.githubusercontent.com/60442877/147976814-ff33ab36-c60b-429d-9302-6bb814b86cdd.png)

### 1.112 Random Forest

Random Forest is an example of ensemble learning, in which we combine multiple machine learning algorithms to obtain better predictive performance.

Why the name “Random”?

Two key concepts that give it the name random:

- A random sampling of training data set when building trees.
- Random subsets of features considered when splitting nodes.

A technique known as bagging is used to create an ensemble of trees where multiple training sets are generated with replacement.

In the bagging technique, a data set is divided into N samples using randomized sampling. Then, using a single learning algorithm a model is built on all samples. Later, the resultant predictions are combined using voting or averaging in parallel.

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

## 2.6 https://www.youtube.com/watch?v=sQ870aTKqiM Random Forest: Missing Data and Clustering

- For the missing data, we firstly have an initial guess for those missing values, then, we need to refine our guess
- We refine the guess by determining which samples are similar to the one with missing data, so we need to determine the similarity first
- Step 1: Build a random forest
- Step 2: Run all of the data down all of the trees and we keep track of similar samples using a 'Proximity Matrix'
- Step 3: We use the proximity values to make better guessses about the missing data
- After we revised our guesses, we do the whole steps again, we do this 6 or 7 times until the missing values converge (i.e. no longer change each time we recalculate)

1 - the proximity values = distance between samples which result in 'Distance Matrix'

## 2.7 https://www.youtube.com/watch?v=LsK-xG1cLYA AdaBoost

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

## 2.8 https://www.youtube.com/watch?v=3CC4N4z3GJc Gradient Boost

- When Gradient Boost is used to predict a continuous value, we say that we are using Gradient Boost for Regression
- Using Gradient Boost for regression is different from doing linear regression
- Gradient Boost starts by making a single leaf instead of a tree or stump, this leaf represents an initial guess of all the samples
- Like AdaBoost, Gradient Boost builds fixed sized trees based on the previous tree's error, but unlike AdaBoost, each tree can be larger than a stump
- Gradient Boost builds another tree based on the errors made by previous tree, and Gradient Boost continuous to build trees in this fashion until it has made the number of trees you asked for, or additional trees fail to improve the fit


Step 1: Build a single leaf which is generated by calculating the average value of whole samples
Step 2: Build a tree based on the errors from the first tree, the errors are called 'Pseudo Residuals'. And we want to build a tree to predict these residual
Step 3: We can combine the orignial leaf with new tree to make a new prediction, thats the average value + learning rate * predicted residual value
Step 4: Build another tree based on the errors made by previous tree, and do prediction again, the prediction value = the average value + learning rate * predicted residual value from first tree + learning rate * predicted residual value from second tree
Step 5: Then, we repeart above steps to make trees until we reach the maximum specified or adding additional trees does not significantly reduce the size of the residuals

Gradient Boost deals with overfitting problem by using a Learning Rate to scale the contribution from the new tree that is used to predict the residuals, and the learning rate is a value between 0 and 1

The final predicted value = average value + learning rate * predicted Residual
