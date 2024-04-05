# News Feed System 

### 1. Problem Formulation
Show feed (recent posts and activities from other users) on a social network platform

* Clarifying questions
  - What is the primary business objective of the system? (increase user engagement?)
  - Do we show only posts or also activities from other users?
  - What types of engagement are available? (like, click, share, comment, hide, etc.)? Which ones are we optimizing for? 
  - Do we display ads as well? 
  - What types of data do the posts include? (text, image, video)?
  - Are there specific user segments or contexts we should consider (e.g., user demographics)?
  - Do we have negative feedback features (such as hide ad, block, etc.)?
  - What type of user-ad interaction data do we have access to can we use it for training our models? 
  - Do we need continual training? 
  - How do we collect negative samples? (not clicked, negative feedback). 
  - How fast the system needs to be? 
  - What is the scale of the system? 
  - Is personalization needed? Yes 
  
* Use case(s) and business goal
  - use case: show friends most engaging (and unseen) posts and activities on a social network platform app (personalized to user)
  - business objective: Maximize user engagement (as a set of interactions)

* Requirements
  - Latency: 200 msec of newsfeed refreshed results after user opens/refreshes the app
  - Scalability: 5 B total users, 2 B daily active users, refresh app twice 
    
* Constraints
  - Privacy and compliance with data protection regulations.
  - hardware constraint.
    
* Data: Sources and Availability:
  - Data sources include user interaction logs, ad content data, user profiles, and contextual information.
  - Historical click and impression data for model training and evaluation.

* Assumptions:
  - Users' engagement behavior can be characterized by their explicit (e.g. like, click, share, comment, etc.) or implicit interactions (e.g. dwell time) 
  
* ML Formulation:
  - Objective: 
    - maximize number of explicit, implicit, or both type of reactions (weighted)
    - implicit: more data, explicit: stronger signal, but fewer data -> weighted score of different interactions: share > comment > like > click etc 
  - I/O
    - I: user_id, O: ranked list of unseen posts sorted by engagement score (wighted sum) 
  - ML Category
    - Ranking problem
      - <mark>point-wise LTR with multi/label (multitask) binary classification</mark>

### 2. Metrics  
* Offline 
  - Multitask ROC AUC for each task (trade-off b/w TPR and FPR)

* Online 
  - <mark>CTR</mark>
  - Reactions rate (like rate, comment rate, etc)
  - Time spent 
  - User satisfaction (survey)

* Non function metric (engineer part)
  - model training time and frequency.
  - generate top k recommend news predict time.
  - cost for training.

### 3. Architectural Components  
* High level architecture
  - prepare data
  - generate train/eval/test data
  - training
    - We can use point-wise learning to rank (LTR) formulation 
    - Options for multi-label/task classification: 
      - Use N independent classifiers (expensive to train and maintain) 
      - Use a multi-task classifier
        - learn multi tasks simultaneously 
        - single shared layers (learns similarities between tasks) -> transformed features 
        - task specific layers: classification heads 
        - pros: single model, shared layers prevent redundancy, train data for each task can be used for others as well (limited data)
  - prediction
  - post-process generate top k news

### 4. Data Collection and Preparation
* Data Sources
  - Doc or query information
    - Users
    - Posts
  - Doc interaction query information
    - User-post interaction 
    - User-user(post author) (friendship)

* Labelling
  - Use user engagement behavior like, click, share, comment as multi-task label.
  - Note: some of the class may be much fewer than others, so label may be unbalanced.

* Data storage (engineer part)
  - User and post information: BQ, cloud storage bucket, etc
  - Interaction information: same as above

* Data generation pipeline (engineer part)
  - Data ingestion
  - Feature generation including user post information and interaction information.
  - Feature transformation like standardized some features or embedding some features.
  - Label generation, create multi class label.

### 5. Feature Engineering

* Feature selection 
  - Posts: 
    - Text
    - Image/videos
    - Number of reactions of each category (likes, shares, replies, etc.)
    - Existing time  
    - Hashtags 
  - User: 
    - ID, username
    - Demographics (Age, gender, location)
    - Context (device, time of day, etc)
    - Interaction history (e.g. user click rate, total clicks, likes rate, shares rate, comments rate, etc. )
  - User-Post interaction: 
    - IDs(user, Ad), interaction type, timestamp, location 
  - User-User(post author) affinities 
    - connection type
    - reaction history (Number of liked/commented/shared rate etc. posts from author)
    - length of friendship. The number of days the user and the author have been friends on the platform.
    - close friends and family, binary value.

* Feature preprocessing / preparation
  - missing feature check
    - choose sample contains all important features, make sure enough training sample, usually 1% train data of predict data
    - features missing above 80%, drop it
    - imputation:
      - Demographics:
        - statistic method
        - from some other model predict value
        
  - feature distribution check
    - numerical
      - var, mean
    - categorical
      - frequency
    - class
      - ratio
  
  - Text: 
    - use a pre-trained LM to get embeddings
    - use BERT here (posts are in phrases usually, context aware helps, and BERT first token contains sentence meaning) 
  
  - Image / Video: 
    - preprocess 
    - use pre-trained models e.g. SimCLR / CLIP to convert -> feature vector 
  
  - Dense numerical: 
    - Engagement feats (Number of clicks, etc.)
      - use directly + scale the range which means normalized numerical data, since different scale will influence model time, accuracy. 
  
  - Discrete numerical: 
    - Age: 
      - bucketize into categorical then one hot, since it's not too sparse
  
  - Hashtags: 
    - tokenize, token to ID, simple vectorization (TF-IDF or word2vec) - no context, need fast and light method

### 6. Model Development and Offline Evaluation

* Model selection 
  - We choose NN
    - unstructured data (text, img, video)
    - embedding layers for categorical features
    - fine tune pre-trained models used for feat eng.
  - multi-labels 
    * P(click), P(like), P(Share), P(comment)
  - Two options: 
    - N NN classifiers
      - Expensive to train. Training several independent DNNs is compute-intensive and time-consuming.
      - For less frequent reactions, there might not be enough training data. This means our system is not able to predict the probabilities of infrequent reactions accurately.
    - Multi task NN (choose this) 
      - Shared layers 
      - Classification heads (click, like, share, comment)
      - Pro:
        - save money and time
        - for some of the task like share this may not have enough to train independently, connect with click together can help reduce model overfitting
        - increase model robust to reduce overfitting by data augmentation with more data
        - some of the task have connection, train together can be information in parameters
        - soft parameter sharing and hard parameter sharing:
          - soft:
            - some of the layers share but each task keep part of independent layer by gate (attention)
          - hard:
            - share all base layers
  - Passive users problem: 
    - All their Ps will be small 
    - Add two more heads 
      * Dwell time (seconds spent on post)
      * P(skip) (skip = spend time < t)
      

* Model training 
  - Loss function: 
    - L = sum of L_is for each task with w_i for each task
      - weight design by manually
      - Uncertainty paper
      - trainable weight parameter
    - for binary classify tasks: CE 
    - for regression task: MAE, MSE, or Huber loss
    - focal loss for some class have few data
  - Dataset 
    - user features, post features, interactions, labels
    - labels: 
      - for multi task use positive, negative for each task (like, didn't like etc)
        - example: like: use like as positive, and impression but not click like as negative, here negative class will much more than positive, keep same ratio by down sampling negative class
      - for dwell time use regression 
    - Imbalanced dataset: down sampling negative 
  - Model eval and HP tuning 
  - Iterations


* Model deploy, save, scale, parallel training or hyperparameters turning (engineer part)
  - deploy on cloud ML platform for easy CD and monitor.
  - save through cloud ML platform.
  - scaling through stochastic mini batch training through trainer component save memory
  - parallel hyperparameters turning through beam or ML platform function.
  
### 7. Prediction Service

* Data Prepare pipeline
  - static features:
    - list text, image, user information do batch feature compute (daily, weekly) -> feature store in cloud
  - dynamic features:
    - number of post clicks, etc -> streaming  

* Prediction pipeline 
  - two stage (funnel) architecture 
    - candidate generation / retrieval service 
      - rule based to shrink size
      - filter and fetch unseen posts by users under certain criteria 
    - Ranking 
      - features -> model -> engagement probability -> sort to get top k news
      - re-ranking: business logic, additional logic and filters (e.g. user interest category) we can do topic modeling for each user.

* Continual learning pipeline 
  - fine tune on new data, eval, and deploy if improves metrics  
  
### 8. Online Testing and Deployment  
* A/B Test 
* Deployment and release (engineer part)
  - deploy platform depends on company ML infrastructure
  - deploy size and cost

### 9. Scaling, Monitoring, and Updates 
* Scaling (SW and ML systems)
  - SW
    - train data partition storage.
    - cache text, image, user information results by weekly or daily, in prediction service.
    - sharding data generation pipeline component like in BQ, pyspark or beam.
  - ML
    - feature generation pipeline parallel
    - <mark>distributed training multi-task classification</mark>
* Monitoring 
  - System gate metrics
    - each component running time and cost
    - system reaction time, latency time
  - ML gate metrics
    - model training time and cost
    - batch prediction time and cost
    - model offline metrics like train and test performance ROC-AUC score save in dashboard or cloud performance tracker
    - model online metrics save on post-process pipeline or prediction system
  - Data gate metrics
    - features distribution like categorical feature unique value cardinality, dense numerical value mean, variance, kl-diverge compare last time training data
    - features correlation
    - each class training sample size ratio
* Updates 
  - model update
    - train from scratch depends on how seriously user information change, this change should be defined early and trigger retrain automatically like alarm.
    - frequency, same as above depends on how seriously user information or interaction change like user's interested topic change. Weekly or monthly.
  - pipeline update
    - depends on monitor, cost tracker or human sense.

### 10. Other topics  
* Viral posts / Celebrities posts
  - should be shown in all connected user news add this business logic in prediction ranking stage
* New users (cold start)
  - rank depends on some rule base criteria 
* Positional data bias
  - point-wise LTR not consider news rank when training
* calibration: 
  - fine-tuning predicted probabilities to align them with actual click probabilities 
* data leakage: 
  - info from the test or eval dataset influences the training process
  - target leakage, data contamination (from test to train set)
* catastrophic forgetting
  - model trained on new data loses its ability to perform well on previously learned tasks, usually happens when new data has big change which should be monitored by data gate metrics