# Friends/Follower recommendation (People you may know)

### 1. Problem Formulation
Recommend a list of users that you may want to connect with

* Clarifying questions:
  - What is the primary business objective of the system? 
  - What's the primary use case of the system?
  - Are there specific factors needs to be considered for recommendations?
  - Are friendships/connections symmetrical?
  - What is the scale of the system? (users, connections)
  - Can we assume the social graph is not very dynamic?
  - Do we need continual training? 
  - How do we collect negative samples? (not clicked, negative feedback). 
  - How fast the system needs to be? 
  - Is personalization needed? Yes 

* Use case(s) and business goal
  - use case: recommend a list of users to connect with on social media app (e.g. facebook, LinkedIn)
  - business objective: maximize number of formed connections 

* Requirements
  - Scalability: 1 B total users, on avg. 10000 connection per user
  - Response to user quickly or extreme accurate.
  
* Constraints
  - Privacy and compliance with data protection regulations.
  - Hardware constraint.
    
* Data
  - Sources and Availability

* Assumptions
  - Symmetric friendships
  
* ML Formulation
  - Objective
    - maximize number of formed connections 
  - I/O
    - I: user_id, O: ranked list of recommended users sorted by the relevance to the user 
  - ML Category
    - Ranking problem: 
      - point-wise LTR: binary classifier (user_i, user_j) -> p(connection)
      - cons: doesn't capture social connections
      - input: user's related behavior or identity features to predict probability of two user will connect
    - Graph representation:
      - edge prediction 
      - pros: supplement with graph structure info (nodes, edges)
      - input: social graph, node features to predict edge b/w nodes 


### 2. Metrics  
* Offline 
  - GNN model: binary classification:
    - ROC-AUC, use as parameters turning
    - Precision-recall, FN is more importance than FP, thus recall is more importance.
  - Recommendation system: binary relationships -> MAP
  
* Online 
  - Number of friend requests sent over X time, for one user top k recommend friends. 
  - Number of friend requests accepted over X time for above same user top k recommendation friends. 

* Non function metric (engineer part)
  - model training time and frequency.
  - generate top k recommend friends predict time.
  - cost for training since GNN usually huge and hard training.


### 3. Architectural Components  
* High level architecture (engineer part)
  - prepare data
  - prepare graph
  - generate train/eval/test data 
  - training
  - prediction
  - post-process generate top k recommend friends (nodes) for each user
  
### 4. Data Collection and Preparation
* Data Sources
  - User information:
    - demographics, edu and work backgrounds, skills, etc
    - note: standardized data (e.g. cs / computer science)
      - Force users to select attributes from a predefined list.
      - Use heuristics to group different representations of an attribute.
      - Use ML-based methods such as clustering or language models to group similar attributes.
        - word2vec -> hdbscan
  - Graph information:
    - User-user connections
    - User-user interactions

* Labelling
  - Graph information, connected as positive, not connected as negative.
  - Note: negative label sampling is importance later for training model.

* Data storage (engineer part)
  - User information: BQ, cloud storage bucket, etc
  - Graph information: same as above

* Data generation pipeline (engineer part)
  - Data ingestion
  - Feature generation including user information and graph information, graph may not need generate.
  - Feature transformation like standardized some features or embedding some features.
  - Label generation, create positive label and negative label edge list.
    - note: in order to prevent over-fitting, negative label list will generate each epoch, which means negative label list number same as epoch number.


### 5. Feature Engineering
* Feature selection
  - User information:
    - ID, username
    - Demographics (age, gender, location)
    - Account/Network info: number of connections, followers, following, requests, etc., account age
    - Interaction history (number of likes, shares, comments) over a certain period, like one week.
    - Context (device, time of day, etc)
  
  - Graph information: 
    - User-user connections: 
      - Connection: IDs(user1, user2), connection type, timestamp, location 
      - Eduction and work affinity: major similarity, companies in common, industry similarity, etc 
      - Social affinity: number mutual connections, profile visits (The number of times a user looks at the profile of another user), time discounted mutual connections
  
    - User-user interactions:  
      - IDs(user1, user2), interaction type, timestamp, location

* Feature preprocessing:
  - Text:
    - preprocess: tokenization, stop word removal, stemming, lemmatization, spelling correction, and encoding
    - use a pre-trained LM to get embeddings like word2vec, BERT
  
  - Image / Video: 
    - preprocess 
    - use pre-trained models e.g. SimCLR / CLIP to convert -> feature vector
  
  - numerical features:
    - Dense:
      - The numbers of connections, followers, following, and pending requests:
        - directly use + scale the range which means normalized numerical data
    - Discrete:
      - not sparse (like less than 20 category):
        - Age: 
          - bucketize into categorical then one hot
      - sparse (more than 100 category):
        - IDs:
          - embedding layer (each ID type its own embedding layer, like in nlp id in here is word)
        
  - Category feature
    - sparse (more than 100 category):
      - City/Country/Language:
        - embedding don't need too many dimension like 5 is ok.
    - not sparse (same as above):
      - Gender
        - one hot
      - Device
        - bucketize into categorical then one hot
      
  - check missing feature number to see if we can use or not.
  - standardized data (e.g. cs / computer science) to make sure no duplicate meaning feature.
  - change numerical data into rate, for example: numer of likes / number of see, or constraint this value in fix time period, because absolute value will keep increasing in the future.
  - encoding some categorical feature, for example: major similarity, defined numerical value show major, minor or etc.
  - analysis graph structure, like average degree, number of node, number of edges, clustering coefficient (how connected node's neighbor nodes) etc. same can add into node feature, or in feature use as graph metric gate alarm.


### 6. Model Development and Offline Evaluation

* Model selection

  - GNN 
    - model task: edge probability prediction
    - modular: GCN -> GraphSAGE -> GAT -> GIT
      - GCN:
        - only apply neighbor information
        - use whole graph to train hard for larger graph
        - weight edge is same for each pair nodes
      - GraphSAGE:
        - apply node itself information
        - inductive way to train, use sample subgraph to train for each node, increase unseen nodes and save time, memory 
        - can add weight for edges
      - GAT:
        - attention method to learn each edge weight
    - hyperparameters: number of layers, hidden states, epoch, dropout rate, linear layer dimension
      - number of layers 2-3 is enough, don't need too far hop
      - epoch with early stopping method
    - regularization:
      - add residual connection for each layer input
      - In each GNN layer BN before activation
      - In each GNN layer add dropout
    - loss function: node embedding to calculate similarities (vector product) between two nodes, mark positive and negative label, use cross-entropy loss.
    - input: graph (node and edge features), node feature from user information, edge feature from user-user information. 
    - output: embedding of each node then use similarities b/w node embeddings for edge prediction, like two vector dot product than sigmoid. 

* Model data prepare
  - Sampling
    - as mention before, negative label random sampling for each epoch.
  - Data splits (train, dev, test)
    - graph train, dev, test split by inductive method. 
  - Class Imbalance
    - balance positive and negative label since we create negative every epoch.

* Model Training 
  - snapshot of G at t. model predict connections at t+1
  - Dataset 
    - create a snapshot at time t
    - compute node and edge features 
    - create labels using snapshot at t + 1 (if connection formed, positive) 
  - Model eval and HP tuning 
  - Iterations 

* Model deploy, save, scale, parallel training or hyperparameters turning (engineer part)
  - deploy on cloud ML platform for easy CD and monitor.
  - save through cloud ML platform.
  - scaling through split large graph into sub-graph and use beam to parallel training and prediction.
    - use GraphSAGE inside GNN each layer, reduce memory and train time
    - use Cluster-GCN to split into subgraph and train
  - parallel hyperparameters turning through beam or ML platform function.
  
### 7. Prediction Service

* Prediction pipeline 
  - Candidate generation 
    - Friends of Friends (FoF) - rule based - from 1B to 1K.1K = 1M candidates -> FoF service.
  - Candidate feature generation by run exactly same feature engineering pipeline when training. 
  - Scoring service by using GNN model -> embeddings -> similarity scores.
  - Generate top k friends by sorting score.
* Batch vs Online prediction
  - Do batch prediction. Pre-compute PYMK tables for each / active users and store in cloud DB.
  - As the social graph in PYMK does not evolve quickly, the pre-computed recommendations remain relevant for an extended period.
  - Update active users feature table since user behavior in graph may change frequently.
* Note: re-rank based on business logic.
  
### 8. Online Testing and Deployment  

* A/B Test
  - use previous metric section mentioned number of friend requests sent over X time, etc
* Deployment and release (engineer part)
  - deploy platform depends on company ML infrastructure
  - deploy size and cost

### 9. Scaling, Monitoring, and Updates

* Scaling (SW and ML systems)
  - SW
    - train data partition storage.
    - cache frequent active user PYMK table results, in prediction service.
    - sharding data generation pipeline component like in BQ, pyspark or beam.
  - ML
    - feature generation pipeline parallel
    - distributed training GNN, split graph into sbu-graph and do parallel training, or other solution see above. 
* Monitoring 
  - System gate metrics
    - each component running time and cost
    - system reaction time, latency time
  - ML gate metrics
    - model training time and cost
    - batch prediction time and cost
    - model offline metrics like train and test performance precision, recall save in dashboard or cloud performance tracker
    - model online metrics save on post-process pipeline
  - Data gate metrics
    - features distribution like categorical feature unique value cardinality, numerical value mean, variance
    - features correlation
* Updates
  - model update
    - train from scratch depends on how seriously graph structure change, this change should be defined early and trigger retrain automatically like alarm.
    - frequency, same as above depends on how seriously graph structure change. Weekly or monthly.
  - pipeline update
    - depends on monitor, cost tracker or human sense.

### 10. Other topics  
* Add a lightweight ranker
  - add inside prediction system pipeline, rule base ranker combine model generate probability ranker, when our cache result not in DB which means for some not active user combine rule base ranker.
* bias problem
  - in negative label random sampling part may introduction bias since negative label may be edge we will predict as positive.
  - combine rule base and random sampling together, rule base select stong candidate negative label and then random sampling.
* delayed feedback problem (user accepts after days)
  - online metric calculate will have bias, so we need fix time range to collect.
* personalized random walk (for baseline)
  - build base model to replace GNN model for scaling issue may happen on GNN.