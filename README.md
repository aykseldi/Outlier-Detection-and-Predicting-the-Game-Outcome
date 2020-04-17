# Outlier-Detection-and-Predicting-the-Game-Outcome

Machine Learning (ML) is truly helpful techniques for prediction and classification. The main objective is to report current progress and experiments’ results from using various ML methods. In this project, we analyze the NBA (National Basketball Association) statistics data for predicting the game outcome and detecting outliers.  Firstly, some feature selection algorithms are implemented and their MSE values are calculated by ANN. NCA-C shows a better performance. Then, some classification algorithms are applied by using NCA-C. SVM shows a better performance whereas Naïve Bayes shows a worse performance. Then we classify the data using two hidden layer (2x2 and 64x64 neurons) pattern recognition network. tansig shows a better performance whereas hardlim shows a worse performance. Secondly, when it comes to outlier detection, our aim is finding the players which break apart from ordinary or standart ones. Our outlier dataset variables are greater than one so we focused on multivariate outliers. In order to find the most contributing features, we used Principle Component Analysis. It helped us extracting decreased dimensional set of features from the features appproximating between 17 to 21. Afterward we practices PyOD tolkit which is specifically designed for outlier detection.  We used 5 algorthms, Isolation Forest, Feature Bagging Detector, KNN, Cluster Based Local Outlier and Average KNN. More or less results of first 4 of them was similar but Average KNN performed worse than others.

Intro

   There are 12 datasets in NBA statistics data [1] and it is used ‘team.season.txt’ for predicting the outcome and ‘player_playoff_career.csv’ was used for outlier detection. At ‘team_season.txt’ each data point is the performance of each team in a season. There are 36 features and 684 information of teams. These data include from 1946 to 2004 seasons. It is used the data beginning from the 1979 season since there are losses and inconsistencies before 1979 seasons. This part is implemented with the MATLAB. On the other hand, at ‘player_playoff_career.csv’ there are 19 features 2055 information is available. Outstanding players were detected based on different features such as points, assists, rebounds etc. Python PyOD library, which provides access to different algorithms for detecting outliers, is made use of during the second part of the work.
   Various approaches for feature selections (SFS, Relieff, and NCA) would be applied along with ANN for team win ratio of per year prediction. For outlier detection, since principle component analysis is fast and flexible unsupervised method, it is used to reduce dimensionality in dataset.
Our report is constructed as follows: (2) report on related work that has been done in the past, (3) proposed methodologies for feature selection, classification and outlier detection model and results from our experiments with those methodologies, (4) conclusion.

Related Work

   There are many work dealing with the topic of predicting the results of sports and detecting the outlier. Bunker and Thabtah [2] proposed a novel SRP-CRISP-DM framework for classification problem. They focused on predicting team sports results. In this work, it is used ANN and match-related features. Miljkovic et al. [3] presented a system for predicting the results of NBA games. It is used Naïve Bayes and multivariate linear regression methods. It is achieved the accuracy was of 67% (a total of 778 games in season 2009/2010). Wang et al. [4] improved the existing method which is a spanning tree-inspired clustering based outlier detection technique. In this paper, they found a new global outlier factor and by using this, they proposed a new minimum spanning tree (MST)-inspired k-NN-based outlier detection method. It is used in both synthetic and real high dimensional data. Their experiment showed that their work was successfully implemented. 
   
       F. Outlier Detection and Outstanding Player Detection

   The dataset is formed from 5 different parts. These are (i) Statistics of players in regular season (19113 records) (ii) Statistics in playoff season (7544 records) (iii) Statistics in all-star season (1463 records) (iv) Statistics in playoff career (2055 records) (v) Statistic in regular season career (3760 records).
   In data exploration, there are 19 features in each dataset and 3 of them are nominal. These features are player’s id, name and last name so these nominal features are not taken into consideration throughout analysis. The remaining features are scaled into 0 to 1 in order to standardize independent variables. 
    Furthermore in data preparation stage, other features in the dataset was comprised of sums of all scores so in order to avoid  most  played player to be a natural outlier, we divide all the scores of players into match played. 
per_match_score= (2) 

   In the case of dimensionality reduction on player_playoff_career data, PCA (Principal Component Analysis) is applied. Because PCA takes lower dimensional set of features from a higher dimensional dataset while capturing as much information as possible. For PCA results of Playoff career dataset is shown below, 
    TABLE 8.  PCA For PLAYER_PLAYOFF_CAREER DATASET

 It show that first 7 principle components captures nearly %95 of  variance in dataset. 
   For the dataset playeoff carees, the dataset stretches from 1949 to 2004 and old datas older than 1980’s lackes values on some features. So we decided to eliminate these data and take only the ones after 1980. PCA results of playeoff season dataset is shown below, 
TABLE9	PCA FOR ALLSTAR DATASET

   For the dataset allstar players, we did not per match calculation since dataset contains only 1 match statistics. PCA results of playeoff season dataset is shown below, 
TABLE10	PCA FOR PLAYEOFF SEASON DATASET

  For outlier detection, Python PyOD library is used which is a scalable library for detecting outliers in multivariate data. Different outlier detection algorithms are used, these are (i) Feature Bagging Outlier Detection (ABOD), (ii) Isolation Forest, (iii) k-Nearest Neighbors Detector, (iv) Cluster-based Local Outlier Detection.and (v) Average KNN
   Feature Bagging Detector: It fits a number of base detectors. It uses base estimator as Local Outlier Factor as default but other estimators such as k-NN or Angle Based Outlier Detector could be user. 
   In playoff career dataset there are 20 outliers and 2036 Inliers. Threshold value is -0.7118109714421073.
   In playoff season  dataset there are 33 outliers and 4430 inliers. Threshold value is -6231948.694033984.


Fig. 5. Distribution of Points with Feature Bagging Detector
   Isolation Forest: This method makes data partitioning by using a set of trees. It calculates an anomaly score examining how much isolated the point in the structure. The anomaly score is then used to identify outliers from normal observations. 
   In playoff career dataset there are 21 outliers and 2034 inliers. Threshold value is -0.14277197028200625.
   In playoff season  dataset there are 45 outliers and 4418 inliers. Threshold value is - - -0.1391610505888.

Fig. 6. Distribution of Points with Isolation Forest Detector
   k-Nearest Neighbors: The distance to its kth nearest neighbor could be viewed as the outlying score.
   In playoff career dataset there are 16 outliers and 2039 inliers.Threshold value is 
-0.07808484312452174.
   In playoff season  dataset there are 34 outliers and 4429 inliers. Threshold value is -0.05731006134932328.

Fig. 7. Distribution of Points with k-NN Detector
   Clustering Based Local Outlier: It classifies the data into small clusters and large clusters. The anomaly score is then calculated based on the size of the cluster the point belongs to, as well as the distance to the nearest large cluster. 
   In playoff career dataset There are 21 outliers and 2034 inliers.Threshold value is 
-0.6599293211770738.
   In playoff season  dataset there are 45 outliers and 4418 inliers. Threshold value is -0.6953386755417442.


Fig. 8. Distribution of Points with Cluster-Based Local Outlier Detector
   Average KNN: It uses the average of all k neighbors as the outlier score..            
   In playoff career dataset There are 10 outliers and 2045 inliers.Threshold value is 
-0.04585606196467415.
   In playoff season  dataset there are 7 outliers and 4456 inliers. Threshold value is - -0.04706805441399679.

We considered a comparison should be made with NBA official website data against our results and methods. In unsupervised learning there is no ground truth for the comparison of results. In order to tackle the issue, we decided to get Most Valued Players of year 2004 from NBA official website. These players are detected with the help of votes from fans, critical assessment of players performance on important matches and other factor like fair player and disipline on matches. 
Here are the list of 2004 MVP players;
TABLE11	NBA MVP OF YEAR 2004 


We compared our results with the official NBA results;
TABLE12	COMPARISON OF 2004 NBA MVP AND OUR RESULTS 

IV. Conclusion

   This project presents a prediction the outcome of NBA games by predicting team performance per season and detection outstanding players per season in NBA. 
   In first experiment for predicting team performance, we define the classes, consider four different feature selection algorithms and having 2, 64 and 128 hidden neurons with tansig activation function and trainlm training function for a single layer feedforward network. NCA-C displays a better performance (least MSE value) among four feature selection algorithms. Then by using NCA-C, four classification algorithm implemented. SVM displays a better performance (80.99%) among four classification algorithm.
In second experiment for predicting team performance, we consider four different neuronal activation functions, trainlm training function, NCA-C feature selection algorithm and two hidden layers with 2x2, and 64x64 neurons for a multi-layer pattern recognition network. tansig displays a better performance whereas hardlim displays a worse performance among four neural activation functions.
While the number of neurons or hidden layers increases, the elapsed time and MSE values also increases. This approach may not be practical.
For detecting outliers in our datasets, we have make use of Python Outlier Detection which is an an open-source Python toolbox for performing scalable outlier detection on multivariate data[7]. 
In feature selection process, we have used PCA and turned its disadvantage of generating principle component without looking at target  into advantage on our unsupervised learning problem. It helped us on reducing dimensionality of data from 19 to 7.
The alghorithms we have used calculated a threshold value for seperating normal ones with the anomalies or outperformer. We used an outlier fraction parameter during our research since our aim was finding observations that are not similar to rest of data. First we took it 0.05 and we have come up with nearly 60 outliers so after some iterations the we hold the value on 0.01.
KNN show greater performance on our dataset and in average its success rate is %87, so it makes him superior in comparison with other algorthms. But it is important to keep in mind that, in unsupervised learning researches it is difficult to detect basic truth. Even though our alghorithms shows %87 success, voting preferences of fans and emotional factors may change MVP voting results.
Future works may include different dataset, may increase the number of activation functions, the number of different training functions, hidden layers or neurons. Also different classification models and network configurations like Bayesian Belief Network, Radial basis neural network etc. may be applied and compared to each other. 

