# 1 - Delta Lake 概念

Delta Lake 是一个开源的存储引擎和事务性处理层，用于在大数据湖中进行高效、可靠和可扩展的数据湖管理。它是由 Databricks 公司推出的，旨在解决大数据湖中数据一致性、可靠性和性能等方面的挑战。

Delta Lake 主要解决了以下几个问题：

1. **数据一致性和事务性**：Delta Lake 支持 ACID（原子性、一致性、隔离性和持久性）事务，可以保证数据的一致性和可靠性。它使用写时复制（copy-on-write）的机制来保证事务的原子性，并提供了事务日志来记录事务的操作历史，以实现数据的回滚和恢复。
2. **数据版本控制**：Delta Lake 支持数据的版本控制功能，可以跟踪数据的历史变化，并允许用户在不同的数据版本之间进行切换和回滚。这使得数据的管理和追溯变得更加简单和可靠。
3. **增量数据处理**：Delta Lake 提供了增量数据处理的功能，可以有效地处理大规模数据湖中的数据更新和变更。它支持基于时间戳的增量更新和合并操作，以及基于条件的更新和删除操作。
4. **数据湖优化**：Delta Lake 提供了一系列优化功能，包括数据索引、数据布局优化、数据统计信息和数据压缩等，可以提高数据湖的查询性能和资源利用率。

总的来说，Delta Lake 是一个用于管理大数据湖的开源存储引擎和事务性处理层，它提供了高效、可靠和可扩展的数据湖管理解决方案，帮助用户更好地管理和分析大规模的数据湖中的数据。

Databricks数据洞察包含以下组件：
- Spark SQL和DataFrames
  Spark SQL是用来处理结构化数据的Spark模块。DataFrames是被列化了的分布式数据集合，概念上与关系型数据库的表近似，也可以看做是R或Python中的data frame。

- Spark Streaming
  实时数据处理和分析，可以用写批处理作业的方式写流式作业。支持Java、Scala和Python语言。

- MLlib
  可扩展的机器学习库，包含了许多常用的算法和工具包。

- GraphX
  Spark用于图和图并行计算的API。

- Spark Core API
  支持R、SQL、Python、Scala、Java等多种语言。

# 2 - Azure Databricks 

创建一个分布式的cluster环境后就可以打开笔记本了，整个环境是在一个workspace中的，虽然自己对Azure接触的不多，但是从公司提供的环境看，接入环境，启动环境集成服务器，以及进入笔记本都非常方便。

但是每次更换笔记本的时候有可能短线，都要重新手动接上cluster，这个要注意。

**Delta数据类型**

- 可以将数据读入 delta table 进行使用。
- 可以对 delta table 进行读写更新操作。
- 可以进行时间旅行，回溯到上一个版本。
- 支持所有的 Pyspark 语法

以下是对各个关键部分内容的补充和代码示例。

## Section 1: Databricks Machine Learning
### Databricks ML
- Identify when a standard cluster is preferred over a single-node cluster and vice versa

在需要高可用性、高吞吐量、可扩展性和数据冗余，需要大规模并行计算，的关键任务场景下，标准集群是更合适的选择。而对于资源有限、非关键任务或开发测试环境，小规模，sklearn任务等，单节点集群则可能更加实用和经济。选择取决于具体的应用需求、资源限制和可用性/复杂度权衡。

- Connect a repo from an external Git provider to Databricks repos.

在workspace，点击创建，即可连接到远程的repo。以下的创建分支，进行commit，pull，push操作都可以在GUI进行。

- Commit changes from a Databricks Repo to an external Git provider.
- Create a new branch and commit changes to an external Git provider.
- Pull changes from an external Git provider back to a Databricks workspace.
- Orchestrate multi-task ML workflows using Databricks jobs.

Databricks中的ML作业也可以像Airflow那样进行task创建和依存关系的建立。使用的是Job cluster，runtime包括Standard（适合ETL）和ML

### Databricks Runtime for Machine Learning
- Create a cluster with the Databricks Runtime for Machine Learning.
- Install a Python library to be available to all notebooks that run on a cluster.

!pip install or install libraries in cluster

有两种进行外部库安装的方式，一种是cluster级别的，这样的安装，可以让所有的笔记本共用库，当然也可以像平常那样，在笔记本级别进行pip安装，这样的安装，可能会有限制范围。

### AutoML
- Identify the steps of the machine learning workflow completed by AutoML.

在GUI左边栏的实验功能。是一种可视化的进行自动机器学习的功能：Experiment

1. 欠损值补全
2. tuning
3. 训练
4. 评价
5. EDA(探索性数据分析)

生成最好的模型后，需要右上角手动登录到register，然后再deploy

- Identify how to locate the source code for the best model produced by AutoML.
```python
from databricks import automl

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

summary = automl.regress(train_df, target_col="price", primary_metric="rmse", timeout_minutes=5, max_trials=10)
```
- Identify which evaluation metrics AutoML can use for regression problems.
  - `help(automl.regress)`
  - https://docs.databricks.com/ja/machine-learning/automl/train-ml-model-automl-api.html

在进行不同类型的机器学习任务时,我们通常使用不同的评估指标来衡量模型的性能。以下是进行回归分析、分类和时间序列预测时常用的一些评估指标:

1. 回归评估指标:
- 均方根误差 (RMSE):衡量预测值与实际值之间的平均误差。
- 平均绝对误差 (MAE): 衡量预测值与实际值之间的平均绝对误差。 
- R-squared (R^2): 解释了模型能够解释数据集中总变化的比例，介于0到1之间，越接近1模型拟合效果越好。

2. 分类评估指标:

- 准确率 (Accuracy): 正确预测的实例数与总实例数的比率。
- 精确率 (Precision): 对于每个类别，正确预测为正的实例数与所有预测为正的实例数的比率。
- 召回率 (Recall): 对于每个类别，正确预测为正的实例数与所有实际为正的实例数的比率。 
- F1分数: 精确率和召回率的调和平均值。
- ROC曲线和AUC: 绘制真正率和假正率曲线，AUC越接近1模型分类性能越好。

3. 时间序列预测评估指标:

- 均方根误差 (RMSE): 衡量实际值与预测值的均方根差。
- 平均绝对误差 (MAE):  实际值与预测值绝对差的平均值。
- 平均绝对百分比误差 (MAPE): 预测误差的绝对值与实际值的比值。
- R-squared (R^2): 解释了模型能够解释数据集中总变化的比例。
- 方向准确率: 模型正确预测方向变化的比例。

- Identify the key attributes of the data set using the AutoML data exploration notebook.
  - 通过查看GUI

### Feature Store
- Describe the benefits of using Feature Store to store and access features for machine learning pipelines.
  - It enables feature sharing and discovery across your organization and also ensures that the same feature computation code is used for model training and inference.
- Create a feature store table.
```python
fs = feature_store.FeatureStoreClient()

## select numeric features and exclude target column "price"
numeric_cols = [x.name for x in airbnb_df.schema.fields if (x.dataType == DoubleType()) and (x.name != "price")]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)

# create fs table and insert records
fs.create_table(
    name=table_name,
    primary_keys=["index"],
    df=numeric_features_df,
    schema=numeric_features_df.schema,
    description="Numeric features of airbnb data"
)

# create and insert
# create fs table
fs.create_table(
    name=table_name,
    # 主键必须
    primary_keys=["index"],
    schema=numeric_features_df.schema,
    description="Original Airbnb data"
)

# insert records later
fs.write_table(
    name=table_name,
    df=numeric_features_df,
    mode="overwrite"
)
```
- Write data to a feature store table.
```python
# overwrite
df_new_feature = numeric_features_df\
  .filter(F.col('index')< 100)\
  .withColumn('new_feature', F.lit(999))

fs.write_table(
    name=table_name,
    df=df_new_feature,
    mode="overwrite"
)

fs.write_table(
    name=table_name,
    df=df_new_feature,
    mode="merge"  # upsert
)

# get_table()とread_table()の違いは押さえておく
feature_table_df = fs.read_table(table_name)
display(feature_table_df)
```

在 Databricks 中`get_table` 和 `read_table` 都是用于读取 Feature Store 中存储的特征表(Feature Table)的方法，但它们存在一些区别:

- `get_table` 是 `FeatureStoreClient` 对象的一个方法，而 `read_table` 是 `FeatureStoreClient` 对象中的 `data_source` 属性的一个方法。
- `get_table` 的语法: `FeatureStoreClient.get_table(name)`
- `read_table` 的语法: `FeatureStoreClient.data_source.read_table(name)`
- `get_table` 返回一个 `FeatureTable` 对象，而 `read_table` 返回一个 Spark DataFrame。
- `get_table` 只能用于读取特征表的元数据和基本信息，而 `read_table` 可以读取特征表的数据并返回 DataFrame，以便进行后续的数据处理和模型训练等操作。
- `get_table` 只需要传入特征表的名称，而 `read_table` 除了传入特征表名称外,还可以传入额外的参数,如 `datetime` 参数来指定读取特定时间戳的特征数据。

```python
# 读取特征表的元数据
feature_table = fs.get_table("my_feature_table")
# 读取特征表的数据
feature_df = fs.data_source.read_table("my_feature_table", datetime="2023-05-01")
```

- Train a model with features from a feature store table.
```python
with mlflow.start_run() as run:
    rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

    # loggingにfsモジュールを使う
    fs.log_model(
        model=rf,
        artifact_path="feature-store-model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=f"feature_store_airbnb_{DA.cleaned_username}",
        input_example=X_train[:5],
        # 模型输入输出的签名(Model Signature)是对机器学习模型的输入和输出数据结构的正式描述。
        signature=infer_signature(X_train, y_train)
    )
```
- Score a model using features from a feature store table.注意这里的score_batch方法。
```python
batch_input_df = inference_data_df.drop("price") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_airbnb_{DA.cleaned_username}/1", 
                                  batch_input_df, result_type="double")
display(predictions_df)
```
### Managed MLflow
- Identify the best run using the MLflow Client API.
```python
# experiment包括很多run，该代码就可以列出所有的实验中的run的列表，按照顺序排列
run_id_best = mlflow.search_runs(
            summary.experiment.experiment_id,
            order_by = ["metrics.val_rmse"]
            )["run_id"][0]

model_uri = f'runs:/{run_id_best}/model'
# PyFuncModelとしてモデルをロード
loaded_model = mlflow.pyfunc.load_model(model_uri)
```
- Manually log metrics, artifacts, and models in an MLflow Run.
```python
with mlflow.start_run(run_name="LR-Log-Price") as run:
    # Take log of price
    log_train_df = train_df.withColumn("log_price", log(col("price")))
    log_test_df = test_df.withColumn("log_price", log(col("price")))

    # Log parameter
    mlflow.log_param("label", "log_price")
    mlflow.log_param("features", "all_features")

    # Create pipeline
    #  R 风格的公式表达式，表示将所有特征变量(除了 price 列)用于预测 log_price 目标变量。
    r_formula = RFormula(
        formula="log_price ~ . - price",
        featuresCol="features",
        labelCol="log_price",
        # 跳过无效数据
        handleInvalid="skip",
    )
    lr = LinearRegression(labelCol="log_price", predictionCol="log_prediction")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(log_train_df)

    # Log model
    mlflow.spark.log_model(
        pipeline_model, "log-model", input_example=log_train_df.limit(5).toPandas()
    )

    # Make predictions
    pred_df = pipeline_model.transform(log_test_df)
    exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))

    # Evaluate predictions
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log artifact
    plt.clf()

    log_train_df.toPandas().hist(column="log_price", bins=100)
    fig = plt.gcf()
    mlflow.log_figure(fig, f"{DA.username}_log_normal.png")
    plt.show()
```
- Create a nested Run for deeper Tracking organization.
  - 两层start_run，提示nested=True。
```python
# Resume the top-level training
with mlflow.start_run(run_id=run_id) as outer_run:
   # Small hack for running as a job
   experiment_id = outer_run.info.experiment_id
   print(f"Current experiment_id = {experiment_id}")

   # Create a nested run for the specific device
   with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
      mlflow.sklearn.log_model(rf, str(device_id))
      mlflow.log_metric("mse", mse)
      mlflow.set_tag("device", str(device_id))

      artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
      # Create a return pandas DataFrame that matches the schema above
      return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                              columns=["device_id", "n_used", "model_path", "mse"])
```
- Locate the time a run was executed in the MLflow UI.
```python
# Notebookの場合
import mlflow

exp_id = ''
runs = mlflow.search_runs(exp_id)
df_runs = spark.read.format("mlflow-experiment").load(exp_id)
display(df_runs)
```
- Locate the code that was executed with a run in the MLflow UI
- Register a model using the MLflow Client API.
```python
from mlflow.tracking.client import MlflowClient
client = MlflowClient()

model_name = f"{DA.cleaned_username}_review"
model_uri = f"runs:/{run_id_best}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# optional
client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using OLS linear regression with sklearn."
)
```
- Transition a model’s stage using the Model Registry UI page.
  - `client.search_model_versions(f"name = '{model_name}'")[0].current_stage`
- Transition a model’s stage using the MLflow Client API.
  - 使用`transition_model_version_stage`
```python
client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)
client.search_model_versions(f"name = '{model_name}'")[0].current_stage
```
- Request to transition a model’s stage using the ML Registry UI page.

## Section 2: ML Workflows
### Exploratory Data Analysis
- Compute summary statistics on a Spark DataFrame using .summary()
  - `display(df.summary())`
- Compute summary statistics on a Spark DataFrame using dbutils data summaries.
  - `dbutils.data.summarize(fixed_price_df)`
- Remove outliers from a Spark DataFrame that are beyond or less than a designated threshold.
  - `display(fixed_price_df.filter(col("price") > threshhold))`
### Feature Engineering
- Identify why it is important to add indicator variables for missing values that have been imputed or replaced.
  - 如果你对分类/数值特征进行任何插补技术，你必须包含一个额外的字段，指定该字段已被插补
  - 即使进行了缺失值插补，添加指示变量也是一种数据预处理的最佳实践，可以最大程度地保留信息、减少偏差、提高模型性能
- Describe when replacing missing values with the mode value is an appropriate way to handle missing values.
  - 使用众数(mode)进行缺失值插补最适合处理类别型特征(categorical features)的缺失值
  - 可以保留数据分布，无需创建新的类别
- Compare and contrast imputing missing values with the mean value or median value.
  - 对于包含大量异常值或离群值的数据集，使用中位数插补会比均值插补更加稳健
  - 如果保留数据分布形状是最重要的，那么均值和中位数插补都是不错的选择,只是中位数插补可能更优
- Impute missing values with the mean or median value.
```python
for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))
imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)
imputer_model = imputer.fit(doubles_df)
# 保留了缺失指示列信息c_na列，并对其他的列进行了变换
imputed_df = imputer_model.transform(doubles_df)
```
- Describe the process of one-hot encoding categorical features.
  - 独热编码将分类变量转换为二进制形式，其中每个类别都被表示为一个新的特征，而每个特征只有一个元素为1，其余为0。
- Describe why one-hot encoding categorical features can be inefficient for tree-based models.
  - 当应用独热编码于基数较高（high cardinality）的分类变量（具有许多不同类别值的变量）时，可能会导致效率降低。特别是在基于树的机器学习方法中（如随机森林或梯度提升），由于虚拟变量化导致连续变量更容易被重视，因此特征的重要性顺序可能变得不清晰，从而可能导致模型性能下降。
### Training
- Perform random search as a method for tuning hyperparameters.
```python
# randome search
# 指定したパラメータ範囲の組み合わせ(e.g. maxDepth:[2, 5, 10 ], numTrees:[5, 10])を指定した探索回数分ランダムに探索し、最も精度(評価指標)が高い組み合わせを採択する方法。
# 違いはGridSearchCVではなく、RandomizedSearchCVを使うこと
from sklearn import svm, datasets, linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.datasets import load_iris

iris = load_iris()
logistic = linear_model.LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
clfrand = RandomizedSearchCV(logistic, distributions, random_state=0)
searchrand = clfrand.fit(iris.data, iris.target)

# grid search
# 指定したパラメータ範囲の組み合わせ(e.g. maxDepth:[2, 5, 10], numTrees:[5, 10])を網羅的に探索し、最も精度(評価指標)が高い組み合わせを採択する方法。
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clfgs = GridSearchCV(svc, parameters)
searchgs = clfgs.fit(iris.data, iris.target)
```

- Describe the basics of Bayesian methods for tuning hyperparameters.
  - 在贝叶斯方法中，我们利用贝叶斯定理从先前获得的组合结果中寻找候选组合，其中某个值的概率变得较小，然后选择具有最高精度（评估指标）的组合。这种方法允许我们基于先前观察到的结果来动态地调整超参数，并且不断地寻找更好的组合，以提高模型的性能。
  - 当我们尝试调整模型的超参数时，传统的方法是通过在一个预定义的参数空间中进行网格搜索或随机搜索来尝试不同的参数组合，然后选择表现最好的组合。这种方法的缺点之一是它需要大量的计算资源和时间，尤其是当参数空间较大时。
  - 贝叶斯方法通过利用之前的试验结果来指导下一次尝试的参数选择，从而在参数空间中寻找到可能更优的区域。具体来说，它使用贝叶斯定理来更新先验概率分布，将先前的试验结果转化为对参数空间的后验概率分布。这意味着在选择下一个参数组合时，更有可能选择那些已经显示出良好性能的区域，而不是随机地在整个参数空间中进行搜索。
  - 在贝叶斯优化的过程中，我们首先根据先验概率选择一个参数组合，然后评估该组合的性能，并利用这个结果来更新参数空间中参数组合的后验概率分布。通过这种迭代过程，我们可以逐步地缩小搜索空间，并集中在可能性能更高的参数组合上。这种方法通常比传统的网格搜索或随机搜索更高效，特别是在高维参数空间或需要大量计算资源的情况下。

- Describe why parallelizing sequential/iterative models can be difficult.
  - 梯度提升（Gradient Boosting）算法是一种迭代的算法，当构建弱学习器（小型模型）时，会使用先前模型的误差。因此，如果试图将处理分布到不同的节点上，节点之间就需要交换误差信息，这就变得很困难。

- Understand the balance between compute resources and parallelization.
  - 要注意到并行化并不是万能的解决方案。有时候，串行算法可能比并行算法更简单、更稳定、更易于理解和维护。因此，在考虑并行化时，需要综合考虑*任务的特性、计算资源的可用性以及额外开销*，并做出合适的决策。

- Parallelize the tuning of hyperparameters using Hyperopt and SparkTrials.
```python
# single-machine hyperopt with a distributed training algorithm (e.g. MLlib)
# SparkMLのモデルでhyperoptを使う場合は以下の通り
num_evals = 4
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))

# distributed hyperopt with single-machine training algorithms (e.g. scikit-learn) with the SparkTrials class.
# sklearnのモデルでhyperoptを使う場合は以下の通り
num_evals = 4
spark_trials = SparkTrials(parallelism=2)
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       trials=spark_trials,
                       max_evals=num_evals,
                       rstate=np.random.default_rng(42))
```

- Identify the usage of SparkTrials as the tool that enables parallelization for
tuning single-node models.
  - 使用SparkTrials作为工具，它能够为调整单节点模型提供并行化功能。
  - SparkTrials是一个工具，用于在Apache Spark中并行化地进行调整单节点模型的超参数搜索。通过使用SparkTrials，可以同时评估多个超参数设置，从而加速整个调参过程。这样可以利用Spark的并行计算能力，在分布式环境中更高效地执行超参数搜索。
  - parallelism，指定了同时评估的最大试验数量。增加parallelism能够允许同时测试更多的超参数设置，从而提高搜索的效率。默认情况下，parallelism的值等于`SparkContext.defaultParallelism`，这个值是Spark上下文中默认的并行度。
### Evaluation and Selection
**Describe cross-validation and the benefits of downsides of using cross-validation over a train-validation split.**
- 交叉验证（cross-validation）及其相对于训练-验证集分割的优缺点。
- 在n折交叉验证中，我们将数据分成n份，其中一份作为验证集，其余的n-1份作为训练集。然后我们重复这个过程n次，每次选择不同的验证集。最终，我们将n次试验的结果取平均值作为最终的性能评估指标。
- 交叉验证的主要优点是它更加稳健和可靠。通过重复多次，我们能够更好地评估模型的性能，减少由于特定数据分割带来的随机性。此外，交叉验证可以更好地利用数据，因为每个样本都被用于了训练和验证，从而更充分地利用了数据信息。
- 然而，交叉验证也有一些缺点。首先，它需要花费更多的计算资源和时间，因为需要重复训练多次。其次，交叉验证在某些情况下可能会导致过拟合，特别是当数据集较小时。另外，对于某些模型，特别是在大数据集上，简单的训练-验证集分割可能已经足够提供良好的性能评估，而不必使用交叉验证。

**Perform cross-validation as a part of model fitting.**
```python
# cvにpipelineを含める場合
# pros: データ漏洩の可能性が低い
# cons: string indexerのようなestimator/transformerがある場合、foldのdatasetに対して毎回変換をかけることになる
stages = [string_indexer, vec_assembler, rf]
pipeline = Pipeline(stages=stages)
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)
cv_model = cv.fit(train_df)

# pipelineにcvを含める場合
# pros: 変換後にfoldのdatasetに分割するため、処理速度向上が見込める
# cons: データ漏洩の可能性がある
cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)
stages_with_cv = [string_indexer, vec_assembler, cv]
pipeline = Pipeline(stages=stages_with_cv)
pipeline_model = pipeline.fit(train_df)
```

**Identify the number of models being trained in conjunction with a grid-search and cross-validation process.**
- 在进行网格搜索（Grid Search）和交叉验证（Cross-Validation）过程中训练的模型数量如何计算的问题。
```python
# パラメータの組み合わせ×foldの数
# 以下の場合だと、(2*2) * 3 = 12回
param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
              .build())
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)   
```

**Describe Recall and F1 as evaluation metrics.**
- Recall（召回率）：召回率是指在所有实际正例中，模型成功预测出的正例的比例。召回率衡量了模型在识别所有真实正例方面的能力，也就是模型能够捕捉到多少真正的正例。召回率的计算公式为：Recall = TP / (TP + FN)，其中 TP 是真正例数，FN 是假负例数。召回率越高，表示模型能够更好地捕捉到正例，但有可能导致更多的假正例。
- F1 分数：F1 分数是精确率（Precision）和召回率的调和平均值，它综合考虑了模型的精确性和召回率。F1 分数的计算公式为：F1 = 2 * (Precision * Recall) / (Precision + Recall)，其中 Precision 表示模型预测的正例中真正的正例比例。F1 分数同时考虑了模型的准确性和全面性，是一个综合评估模型性能的指标。F1 分数越高，表示模型在精确性和召回率之间取得了更好的平衡。
- 召回率更侧重于模型对正例的覆盖能力，而 F1 分数则更全面地评估了模型的性能，包括精确性和全面性。在某些场景下，召回率可能更为重要，例如在医学诊断中需要尽可能少地漏诊；而在其他场景下，如垃圾邮件过滤，精确性和全面性都很重要，因此 F1 分数可能更为合适。
- Identify the need to exponentiate the RMSE when the log of the label variable is used.
- Identify that the RMSE has not been exponentiated when the log of the label variable is used.
```python
# 目的変数の分布が歪んでいるときに、logをとって正規分布に近づけることでモデルの精度が向上する場合がある
# RMSEはrootをとって単位を合わせるので、正しくRMSEを解釈するために、logではなく、実数に戻す必要あり。そのときにexponentiateする

log_train_df = train_df.withColumn("log_price", log(col("price"))) #学習データ
log_test_df = test_df.withColumn("log_price", log(col("price"))) #テストデータ

r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip") 

lr.setLabelCol("log_price").setPredictionCol("log_pred")
pipeline = Pipeline(stages=[r_formula, lr])
pipeline_model = pipeline.fit(log_train_df)
pred_df = pipeline_model.transform(log_test_df)

#exponentiateしない場合
exp_df_noexp = pred_df.withColumn("prediction", col("log_pred"))

regression_evaluator_noexp = RegressionEvaluator(labelCol="log_price", predictionCol="prediction")
rmse_noexp = regression_evaluator.setMetricName("rmse").evaluate(exp_df_noexp)
print(f"RMSE is {rmse_noexp}")

#exponentiateする場合
exp_df = pred_df.withColumn("prediction", exp(col("log_pred")))

rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
print(f"RMSE is {rmse}")
```
- 这段内容涉及在使用标签变量的对数（log）时需要对 RMSE 进行指数化（exponentiate）的原因。
- 首先，当目标变量的分布是偏斜的（即不符合正态分布）时，有时将其取对数可以使其更接近于正态分布，这有助于提高模型的预测精度。
- 在这个例子中，训练数据集和测试数据集中的价格标签被取对数，以使其更接近于正态分布。然后，一个线性回归模型被用于预测对数价格（log_price）。
- 然而，在评估模型的性能时，我们通常希望使用原始单位进行比较，因此需要将预测结果从对数空间转换回原始单位。这就是为什么在预测数据中创建了一个新列，其中将对数预测（log_pred）指数化为实际的价格预测值。
- 最后，分别计算了未指数化和指数化后的 RMSE，以评估模型的性能。指数化是为了确保 RMSE 能够反映出模型在原始数据空间中的预测误差，以便更好地理解和解释模型的性能。
## Section 3: Spark ML
### Distributed ML Concepts
**Describe some of the difficulties associated with distributing machine learning models.**
- 分布式机器学习模型的困难

1. 数据分布：
   - 异质数据：数据通常分布在多个节点上，确保所有相关数据点可用于训练是一个挑战。
   - 同步问题：从不同节点聚合数据需要同步，这可能消耗大量资源和时间。

2. 模型一致性：
   - 参数同步：在不同节点之间确保模型参数的一致性，特别是在异步环境中，这是一个挑战。
   - 版本控制：管理不同版本的模型并确保所有节点使用正确的版本可能很困难。

3. 计算负载：
   - 资源分配：有效地分配计算负载以避免瓶颈是复杂的。
   - 可扩展性：确保模型能有效地随数据量和节点数量的增加而扩展。

4. 通信开销：
   - 网络延迟：高网络延迟会减慢训练过程。
   - 数据传输成本：在节点之间传输大量数据既昂贵又缓慢。

5. 容错性：
   - 节点故障：处理节点故障而不中断训练过程是关键但具有挑战性。
   - 数据丢失：在节点故障期间确保数据不丢失需要强大的备份和恢复机制。

6. 超参数调优：
   - 复杂性：在分布式环境中调优超参数增加了协调和同步的复杂性。
   - 资源密集性：在分布式节点上运行多个超参数调优迭代会消耗大量资源。

7. 算法设计：
   - 适应性：并非所有的机器学习算法都容易适应分布式框架。
   - 优化：设计能在分布式环境中高效运行且不显著降低性能或精度的算法是个挑战。

- 例子：Spark中的max_bins
  - 在Spark中，数据按行分布在多个工作节点上。每个工作节点需要计算每个特征在每个分割点上的统计信息，并聚合这些统计信息来决定分割点。
  - 挑战：如果Worker1有一个其他工作节点没有的唯一值（例如，32），很难确定这是一个好的分割点。
  - 解决方案：Spark使用maxBins参数将连续变量离散化为桶。然而，桶的数量必须与基数最高的分类变量相同，这增加了复杂性。

**Identify Spark ML as a key library for distributing traditional machine learning work.**
**Identify scikit-learn as a single-node solution relative to Spark ML.**

- Spark ML和scikit-learn的对比
  - Spark ML：关键库，用于分布式传统机器学习工作。
  - scikit-learn：单节点解决方案，相对于Spark ML。
- 将sklearn代码迁移到Databricks时：
  - 直接在多节点ML集群上运行sklearn代码不会提高处理速度。
  - 原因：sklearn假定单节点，不支持分布式处理。
  - 需要重构为Spark ML和Spark DataFrame。

### Spark ML Modeling APIs
**Split data using Spark ML**
```python
# SparkMLの場合
train_df, test_df = df.randomSplit([.8, .2], seed=42)

# sklearnの場合
from sklearn.model_selection import train_test_split
X = df.select([pair[0] for pair in df.dtypes if pair[0] != 'price']).toPandas()
y = df.select(['price']).toPandas()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```
**Identify key gotchas when splitting distributed data using Spark ML.**
- Spark ML分布式数据拆分的关键注意事项
- 保持再现性：固定seed值，即使会话改变也能保持相同的随机分布。
- 重新分区影响：使用repartition可能导致再现性丧失。
- 以下内容得出repartition前后的count是不一样的。
```python
print(f'count before repartition: {train_df.cache().count()}')
train_df_repartition, test_df_repartition = df.repartition(24).randomSplit([.8, .2], seed=42)
print(f'count after repartition: {train_df_repartition.cache().count()}')
```

**Train / evaluate a machine learning model using Spark ML.**
```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
# 特征集合器 / 并对训练数据和测试数据进行变换
vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")
vec_train_df = vec_assembler.transform(train_df)
vec_test_df = vec_assembler.transform(test_df)
# 创建训练模型实例 / 进行拟合
lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(vec_train_df)
# 使用模型，对测试数据进行预测
pred_df = lr_model.transform(vec_test_df)
# 回归评估器实例 / 评估结果
regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
```
**Describe Spark ML estimator and Spark ML transformer.**
- 推定器（estimator）：一种算法，它根据DataFrame中的数据进行拟合，并生成一个转换器（Transformer）。例如，学习算法从DataFrame中学习并生成模型就是一种推定器。推定器具有.fit()方法，用于从DataFrame中学习（或“拟合”）参数。
- 转换器（transformer）：将一个DataFrame转换为另一个DataFrame。它接收一个DataFrame作为输入，并返回一个添加了一个或多个列的新DataFrame。转换器不从数据中学习参数，只是简单地应用基于规则的转换。转换器具有.transform()方法。

**Develop a Pipeline using Spark ML.**
- 在Spark ML中，One Hot Encoder在对类别特征进行编码之前需要先进行字符串索引（String Indexer）变换，这是因为One Hot Encoder只能处理数值类型的数据，而不能直接处理字符串类型的数据。
```python
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer
# 数据分割
train_df, test_df = df.randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)

stages = [string_indexer, ohe_encoder, vec_assembler, lr] #estimatorをリスト化する
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df) #まとめてfitして、transformerを作成する

# optional
pipeline_model.write().overwrite().save(DA.paths.working_dir) #transformerに変換したパイプラインをまるごと保存
saved_pipeline_model = PipelineModel.load(DA.paths.working_dir) #読み込み

pred_df = saved_pipeline_model.transform(test_df) #指定した順番通りにまとめてtransform（transformerなのでtransformメソッド持っている）
```
**Identify key gotchas when developing a Spark ML Pipeline.**
### Hyperopt
- Identify Hyperopt as a solution for parallelizing the tuning of single-node models.
- Identify Hyperopt as a solution for Bayesian hyperparameter inference for distributed models.
- Parallelize the tuning of hyperparameters for Spark ML models using Hyperopt and Trials.
- Identify the relationship between the number of trials and model accuracy.
### Pandas API on Spark
**Describe key differences between Spark DataFrames and Pandas on Spark DataFrames.**

**Dataframe有以下三种类型**
- 1: pandas dataframe
  - 在数据科学家中最常见
  - 可变（可修改），即时执行，保留行的顺序
  - 优点：在数据集较小的情况下性能非常高
  - 缺点：假设在单个节点上运行，数据集较大时会发生内存溢出（OOME）
  - 通常，数据科学家用pandas创建数据，工程师为了实际运行会重构成spark
- 2: spark dataframe
  - 分布式，延迟计算，不可变，不保留行的顺序
  - 优点：在大规模数据情况下性能非常高
  - 缺点：与pandas的方法不兼容
- 3: pandas API on spark
  - 性能接近spark（严格来说，spark > pandas API on spark），操作方式接近pandas，兼具两者的优点

- Identify the usage of an InternalFrame making Pandas API on Spark not quite as fast as native Spark.

**Pandas api on spark在后台管理internal frame（Spark dataframe和元数据）。**

- 仅更新元数据的情况
  - 当指定列为索引时，不需要更新后台的spark dataframe，只需更新元数据即可。
  - 在这种情况下，只更新internal frame的元数据状态。
- 更新spark dataframe的情况
  - 当添加列时（例如，psdf['x2'] = psdf.x * psdf.x），需要同时更新元数据和数据。
  - 在这种情况下，需要更新internal frame的元数据状态和dataframe本身。
  - 以inplace方式更新时，不返回新的dataframe，而是更新内部数据的状态。

- Identify Pandas API on Spark as a solution for scaling data pipelines without much refactoring.
pandasのお作法と似ているため、ソースコードの修正は最小限で分散処理の恩恵を受けることができる

- Identify how to import and use the Pandas on Spark APIs
- Convert data between a PySpark DataFrame and a Pandas on Spark DataFrame.
```python
# 読み込み方法
# spark df
spark_df = spark.read.parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")

# pandas df
import pandas as pd
pandas_df = pd.read_parquet(f"{DA.paths.datasets.replace('dbfs:/', '/dbfs/')}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")

# pandas api on spark
import pyspark.pandas as ps
psdf = ps.read_parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")


# 変換方法
# spark df => pandas df
pandas_df = spark_df.toPandas()
print(f'spark df => pandas df: {type(pandas_df)}')

# spark df <= pandas df
spark_df = spark.createDataFrame(pandas_df)
print(f'spark df <= pandas df: {type(spark_df)}')

# spark df => pandas api on spark
psdf = spark_df.to_pandas_on_spark()
psdf = ps.DataFrame(spark_df)
print(f'spark df => pandas api on spark: {type(psdf)}')

# spark df <= pandas api on spark
spark_df = psdf.to_spark()
print(f'spark df <= pandas api on spark: {type(spark_df)}')

# pandas df => pandas api on spark
from pyspark.pandas import from_pandas
psdf = from_pandas(pandas_df)
print(f'pandas df => pandas api on spark: {type(psdf)}')

# pandas df <= pandas api on spark
pandas_df = psdf.to_pandas()
print(f'pandas df <= pandas api on spark: {type(pandas_df)}')
```
### Pandas UDFs/Function APIs
- Identify Apache Arrow as the key to Pandas <-> Spark conversions.
- Describe why iterator UDFs are preferred for large data.
- Apply a model in parallel using a Pandas UDF.
- Identify that pandas code can be used inside of a UDF function.
- Train / apply group-specific models using the Pandas Function API.
## Section 4: Scaling ML Models
### Model Distribution
- Describe how Spark scales linear regression.
- Describe how Spark scales decision trees.
### Ensembling Distribution
- Describe the basic concepts of ensemble learning.
- Compare and contrast bagging, boosting, and stacking

## 学习笔记本补充参考内容

#### **数据清洗 Data Cleansing**

- `df.describe()` 和 `df.summary()` 是对spark dataframe的统计描述，`summary()` 比 `describe()` 增加了四分位数的描述。
- `dbutils.data.summarize(df)` 可以对数据进行更详细的统计分析，最后一列是一个可视化图表，非常耳目一新。
- 在 Spark ML（Spark机器学习）中，数值通常被表示为 double 类型。这主要是因为 double 类型在存储精度和数值范围上都比较适合机器学习中的数据表示和计算。所以在数据清洗的时候要将整数 `integerType()` 进行 `.cast('double')` 的处理。
- 缺失值标记：

```python
for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))
```

- *Transformers* 是一类用于将数据集进行转换的对象。它们接受一个 DataFrame 作为输入，并生成一个新的 DataFrame 作为输出。常见的转换操作包括特征处理（如特征提取、特征转换、特征选择等）、数据清洗、数据规范化等。主要通过 `transform()` 方法来进行转换操作。（注意，它的变换不是基于学习，而是基于规则。）
- *Estimators* 是一类用于训练模型的对象。它们接受一个 DataFrame 作为输入，并返回一个模型（Model）对象。通常，Estimators 是通过对输入数据进行学习（即训练）来生成模型的。Estimators 主要通过 `fit()` 方法来进行训练操作。
- 在 Spark MLlib 中，Transformers 和 Estimators 通常被组合使用，构建成一个数据处理和建模的流水线（Pipeline）。这种流水线的设计使得用户可以将数据处理和建模过程整合在一起，并且可以很方便地将不同的数据处理步骤和建模步骤组合起来，形成一个完整的数据处理和建模流程。

```python
from pyspark.ml.feature import Imputer
imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)
imputer_model = imputer.fit(doubles_df)
imputed_df = imputer_model.transform(doubles_df)
# 将清洗好的数据存入 delte，就可以进行下一步的模型训练
imputed_df.write.format("delta").mode("overwrite").save(f"{DA.paths.working_dir}/imputed_results")
```

#### **线性回归 Linear Regression**

- 独特的数据分割方法，对训练集和测试集进行*分区*并进行*分割*。（传统我们用的都是sklearn的 `model_selection` 的 `train_test_split`），下面是代码：
```python
train_repartition_df, test_repartition_df = (airbnb_df
                                             .repartition(24)
                                             .randomSplit([.8, .2], seed=42))

print(train_repartition_df.count())
```

- 数据预测：使用 `transform` 方法进行预测。方法如下：
```python
vec_test_df = vec_assembler.transform(test_df)
pred_df = lr_model.transform(vec_test_df)
pred_df.select("bedrooms", "features", "price", "prediction").show()
```
- 模型评价方法也是需要一个实例的：
```python
regression_evaluator = RegressionEvaluator(
    predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")
```
- categorical 数据类型的处理：这里需要进行两次转换，*从字符串到索引，再从索引到独热编码*。

```python
from pyspark.ml.feature import OneHotEncoder, StringIndexer
categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)
```

- `VectorAssembler`是一个可以把多个特征整合为一个大的向量矩阵的实例。其实它的实际功能相当于，讲一个dataframe格式的数据，转化为一个numpy数据，或者一个tensor数据。比如有一组数据如下：
```
+---+---+
|  x|  y|
+---+---+
|  1|  2|
|  3|  4|
|  5|  6|
+---+---+
```
可以转化为：
```
+---+---+---------+
|  x|  y|features |
+---+---+---------+
|  1|  2|[1.0,2.0]|
|  3|  4|[3.0,4.0]|
|  5|  6|[5.0,6.0]|
+---+---+---------+
```
notebook的代码如下：

```python
from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = ohe_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
```
- PipeLine：以上所有的过程，加上建模，都可以融合进一个pipeline中一下子执行。

```python
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol="price", featuresCol="features")

from pyspark.ml import Pipeline
stages = [string_indexer, ohe_encoder, vec_assembler, lr]
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train_df)
```

- 模型预测和评价：
```python
pred_df = saved_pipeline_model.transform(test_df)
display(pred_df.select("features", "price", "prediction"))

from pyspark.ml.evaluation import RegressionEvaluator
regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
```

#### **MLFlow**

- 主要是解决实验追踪困难，代码再现困难，模型打包和部署没有标准化的问题。
- 使用`mlflow.set_experiment()`设置实验。一组`experiment`可以管理多个`run`单位。每一个`run`可以保存参数，代码，指标，输出文件，日志等内容。
- 注意：Spark的模型的话，MLflow只能记录`PipelineModels`的日志。
- 下面是一个完整的工作流代码：

```python
import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="LR-Single-Feature") as run:
    # Define pipeline
    vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="price")
    pipeline = Pipeline(stages=[vec_assembler, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log parameters
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "bedrooms")

    # Log model
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas()) 

    # Evaluate predictions
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regression_evaluator.evaluate(pred_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
```
- 在运行结束后可以在 experiment tab栏找到可视化的操作界面分析刚刚输出的结果。这个很像是AWS的AutoML的自动实验。
- 只要通过上述方式不断使用`with mlflow.start_run(run_name="LR-Single-Feature") as run`构建新的run task，就可以将各种结果进行记录和比较。
- 使用以下代码列出所有的实验run。
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.list_experiments()
```
- 使用`search_runs`可以检索到所有的run实验对象。`run`本身是一个输出结果的list，`run.info`,`run[index].info`,`run[index].data`都可以输出实验结果。

#### **MLflow Model Registry**

- 是MLflow的一个组件，用于管理机器学习模型的生命周期。它提供了一种集中式的方式来跟踪、共享、审核和部署机器学习模型，以便团队中的不同成员可以协作管理模型的版本、权限和部署。
- 版本控制功能：轻松跟踪模型的演进和改进。
- 协作与共享：不同成员可以共享和访问注册表中的模型，并且可以查看模型的元数据、评估指标和文档。这样可以加强团队之间的协作和知识共享。
- 权限管理：允许管理员设置不同用户对模型的访问权限，包括查看、注册、转换、删除等操作。
- 模型审核：CICD流程，管理员可以审核模型版本，并且可以添加审批注释，以便其他团队成员了解模型的审核状态和理由。
- 模型部署：注册表中的模型可以直接部署到不同的环境中，如生产环境、测试环境等。MLflow提供了与各种部署目标（例如MLflow Serving、Azure ML、Amazon SageMaker等）集成的能力。
- 可扩展性：是一个灵活的平台，允许用户根据自己的需求进行扩展和定制。你可以根据团队的规模和需求，定制注册表的工作流程和规则。
- 通过`mlflow.register_model()`可以登录模型。
- 可以通过GUI界面看到自己保存的模型。
- 通过使用`MlflowClient()`可以对模型注册表进行各种操作：`get_model_version()`,`update_registered_model()`可以更新模型说明等。
- 模型部署以及模型状态迁移，一共有四个stage状态：None，Staging（用于模型测试），Production（测试评价后部署到生产环境），Archived。
  - 通过`transition_model_version_stage()`进行stage的迁移。
- 模型载入方法：`mlflow.pyfunc.load_model(model_version_uri)`。
- 使用载入的模型进行预测：`model_version_1.predict(X_test)`。
- 删除模型：`delete_model_version()`，删除全体模型`delete_registered_model()`。

#### **Single Decision Trees**

- 决策树算法中不需要对字符串特征进行OHE操作。
- 决策树算法特征重要度：`dt_model.featureImportances`。是对打包前的特征量的重要度一览，但是重要度为0的特征会不显示。使用以下代码 Pandas 将其变成可读的 dataframe。
```python
features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), dt_model.featureImportances)), columns=['feature', 'importance'])
```
- 以上可以得出大部分特征重要度为 0 ，是因为参数 maxDepth 默认设置为5了，只有 5 个特征量被使用了。
- 决策树具有尺度不变性（scale invariant），就算改变数据的尺度，对决策也不会有很大影响。（见概念补充部分）

#### **Random Forests and Hyperparameter Tuning**

- 为随机森林进行超参调优设置网格搜索的代码。
```python
from pyspark.ml.tuning import ParamGridBuilder
param_grid = (
   ParamGridBuilder()
   .addGrid(rf.maxDepth, [2, 5])
   .addGrid(rf.numTrees, [5, 10])
   .build()
)
```
- 为随机森林进行交叉验证的代码。
```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
evaluator = RegressionEvaluator(
   labelCol='price',
   predictionCol='prediction'
)
cv = CrossValidator(
   estimator=pipeline, # 使用哪个模型
   evaluator=evaluator, # 使用什么进行评价
   estimatorParamMaps=param_grid, # 使用什么超参
   numFolds=3,
   seed=42,
)
cv_model = cv.fit(train_df) # 这个跑的很慢
```
- 使用并行计算跑 fit 会提升一点：
```python
cv_model = cv.setParallelism(4).fit(train_df)
```
- 以上为止都是在 pipeline 中进行 cv 操作，反过来在 cv 中也可以加入 pipeline。前者更安全但是开销更大，每一步都会被评估，后者虽然减少了代码重复，但是有可能会数据泄露（因为某些特征处理或数据预处理步骤可能使用了整个训练数据集）。使用交叉验证和 Pipeline 的组合通常能够更准确地评估模型的性能，但在使用时需要注意计算开销和潜在的数据泄露问题。选择合适的方法取决于具体的问题和数据集特征。
```python
cv = CrossValidator(
   estimator=rf, 
   evaluator=evaluator, 
   estimatorParamMaps=param_grid, 
   numFolds=3, 
   parallelism=4, 
   seed=42
)
stages_with_cv = [string_indexer, vec_assembler, cv]
pipeline = Pipeline(stages=stages_with_cv)
pipeline_model = pipeline.fit(train_df)
```

#### **Hyperopt**

Hyperopt 是一个用于超参数优化的 Python 库，旨在帮助用户自动地搜索最佳的超参数组合，以优化机器学习模型的性能。Hyperopt 提供了多种优化算法和搜索空间定义方式，能够有效地探索超参数空间，并发现最优的超参数配置。

Hyperopt 的主要特点包括：

1. **支持多种优化算法：** Hyperopt 实现了几种流行的优化算法，包括随机搜索（Random Search）、序贯模型优化（Sequential Model-Based Optimization，如 TPE 算法）等。用户可以根据自己的需求选择合适的算法。

2. **可扩展性：** Hyperopt 提供了灵活的接口和API，允许用户自定义搜索空间和目标函数，以满足各种优化任务的需求。

3. **并行化支持：** Hyperopt 支持并行化优化，可以利用多个 CPU 核心或分布式计算资源加速超参数搜索过程。

4. **集成现有机器学习库：** Hyperopt 可以与多种现有的机器学习库（如 scikit-learn、TensorFlow、PyTorch 等）集成使用，使得用户可以在其喜爱的框架中应用超参数优化技术。

使用 Hyperopt 进行超参数优化的一般步骤如下：

1. **定义搜索空间：** 用户需要定义待优化的超参数空间，可以是连续值、离散值或条件变量等。

2. **定义目标函数：** 用户需要定义一个评估目标函数，该函数接受超参数作为输入，并返回一个用于评估性能的指标（如准确率、均方误差等）。

3. **选择优化算法：** 根据任务的性质和需求，选择适合的优化算法（如随机搜索、TPE算法等）。

4. **运行优化：** 使用选定的优化算法运行超参数优化过程，以搜索最佳的超参数组合。

5. **评估结果：** 对于找到的最佳超参数组合，进行模型训练和性能评估，以验证优化结果的有效性。

Hyperopt 的灵活性和易用性使其成为了许多机器学习从业者和研究人员喜爱的超参数优化工具之一。

代码实例（airbnb的pipeline示例参见github链接）：

```python
from hyperopt import fmin, tpe, hp, Trials
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 定义超参数搜索空间
space = {
    'n_estimators': hp.choice('n_estimators', range(10, 1000)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 1),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
}

# 定义目标函数
def objective(params):
    clf = RandomForestClassifier(**params)
    score = cross_val_score(clf, X, y, cv=5).mean()
    return -score  # Hyperopt 默认最小化目标函数，因此加负号使其最大化

# 运行超参数优化
trials = Trials()
best_params = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

# 输出最佳参数
print("Best Parameters:", best_params)
```

#### **AutoML**

- 简化整个机器学习流程的服务，可以对模型进行自动训练和调优，可以在UI环境执行和查看，还可以对实验结果进行解释。
```python
from databricks import automl
# 创建 AutoML 实例
aml = automl.Automl()
# 设置特征工程选项
aml.set_feature_options(strategy="auto", categorical_features=["feature1", "feature2"])
# 运行 AutoML
aml.fit(data, target_column)
# 获取最佳模型
best_model = aml.best_model
# 获取最佳超参数
best_params = aml.best_params
# 进行模型解释性分析
explanation = aml.explain_model(best_model)
# 将最佳模型部署到生产环境
aml.deploy_model(best_model)
```
- 另一个更特定的例子：
```python
from databricks import automl
summary = automl.regress(train_df, target_col="price", primary_metric="rmse", timeout_minutes=5, max_trials=10)
# 打印最好的实验结果
print(summary.best_trial)
```

#### **Feature Store**

- 是一个存储特征的仓库，字面意思。通过以下代码也可以进行定义。
```python
from databricks import feature_store
fs = feature_store.FeatureStoreClient()
```

- 创建特征表：
```python
fs.create_table(
    name=table_name,
    primary_keys=["index"],
    df=numeric_features_df,
    schema=numeric_features_df.schema,
    description="Numeric features of airbnb data"
)
```

然后就可以在特征量仓库中找到各种特征了。

通过代码也可以确认特征：
```python
fs.get_table(table_name).path_data_sources
fs.get_table(table_name).description
```

除此之外还有可以更新特征和记录日志。以及可视化表示。

#### **XGBoost**

可以使用第三方的库进行训练。`from xgboost.spark import SparkXGBRegressor`，然后作为pipeline的一部分进行训练。

#### **Inference with Pandas UDFs**

PandasUDF（Pandas User Defined Function）是 Apache Spark 中的一种用户自定义函数，用于在 PySpark 中执行基于 Pandas 的操作。PandasUDF 允许用户编写自定义函数，这些函数以 Pandas 数据帧（DataFrame）作为输入，并返回 Pandas 数据帧作为输出。在执行过程中，Spark 会自动将数据分割为多个分区，并在每个分区上执行自定义函数，最后将结果合并起来。

并且它利用Apache Arrow使得计算高速化。

Apache Arrow是一个跨语言的内存布局和数据传输格式，旨在提高大规模数据分析的性能和互操作性。它提供了一个统一的内存数据结构，用于在不同的系统和编程语言之间高效地传输和共享数据。它支持多种编程语言，包括 Python、Java、C++、R 等，可以在这些语言之间高效地传输和共享数据。

Arrow 提供了一种内存布局格式，以最大程度地减少数据传输和序列化开销。它使用了列式存储和扁平内存布局，使得数据可以被快速加载到内存中，并且易于进行高效的数据操作。支持零拷贝操作，可以在不同的系统和编程语言之间高效地传输数据，而无需复制或序列化数据。这大大提高了数据传输的速度和效率。

Apache Arrow 提供了统一的数据格式和接口，使得不同系统和应用程序之间可以轻松地共享和交换数据。它可以与多种开源工具和项目集成，如 Apache Spark、Pandas、NumPy 等。

# 3 - Pyspark 学习笔记

打印schema：
```python
train_data.printSchema()
```

对Categorical数据进行处理：
```python
from pyspark.ml.feature import (VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer)
gender_indexer = StringIndexer(inputCol='Sex',outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex',outputCol='SexVec')
```

训练fit数据往往不是sklearn那样包含X，y，而是将两者打包了：
```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["Avg Session Length", "Time on App", "Time on Website", 'Length of Membership'], outputCol="features")
output = assembler.transform(data)
final_data = output.select("features",'label')
train_data, test_data = final_data.randomSplit([0.7, 0.3])
lrModel = lr.fit(train_data)
```

进行结果预测：
```python
final_results = final_lr_model.transform(test_new_data)
final_results.select('id','prediction').show()
```

用evaluate进行评估：
```python
pred_and_labels = fitted_model.evaluate(test_data)
```

显示结果：
```python
pred_and_labels.predictions.show()
```

使用二分类evaluator进行结果auc结果评估：
```python
label_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')
auc = label_eval.evaluate(pred_and_labels.predictions)
```

多分类结果评估：
```python
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='accuracy')
```

Pipeline自身就可以看作是一个model：
```python
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[gender_indexer, embark_indexer, gender_encoder, embark_encoder, assembler, log_reg_titanic])
fit_model = pipeline.fit(train_titanic_data)
results = fit_model.transform(test_titanic_data)
```

梯度提升树官方sample：
```python
from pyspark.ml.classification import GBTClassifier
data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
(trainingData, testData) = data.randomSplit([0.7, 0.3])
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)
model = gbt.fit(trainingData)
predictions = model.transform(testData)
predictions.select("prediction", "label", "features").show(5)
```

一个印象深刻的项目：在狗狗食品中4中防腐剂哪种对快速腐败具有重要影响。这道题的目的是找出features中的最重要的feature。

repo地址：machine-learning-lab/Pyspark/Spark_for_Machine_Learning/Tree_Methods/Tree_Methods_Consulting_Project.ipynb

# 4 - 知识概念补充

**优化参数方法 TPE**

TPE（Tree-structured Parzen Estimator）是一种用于优化超参数的算法，特别适用于基于贝叶斯方法的优化框架。TPE 算法由James Bergstra 和Yoshua Bengio 在2011年提出。

TPE 算法的核心思想是将优化问题转化为一个概率密度估计的问题，通过对超参数的先验概率密度和条件概率密度的估计来搜索出最优的超参数。与传统的随机搜索或网格搜索不同，TPE 算法在搜索空间中不断地更新先验概率密度和条件概率密度，并根据这些概率密度来选择下一次尝试的超参数。

具体来说，TPE 算法分为两个阶段：

- 建模阶段（Modeling Phase）：在这个阶段，TPE 算法使用贝叶斯方法来建模目标函数与超参数之间的关系。首先，对于每个超参数，TPE 算法分别建模其先验概率密度，通常假设为一些特定的概率分布（如均匀分布、高斯分布等）。然后，TPE 算法根据目标函数的表现，计算每个超参数的条件概率密度，即给定目标函数的表现，每个超参数的可能取值的概率分布。最后，TPE 算法利用先验概率密度和条件概率密度来计算一个指标（Expectation Over Improvement，EOI），用于衡量选择某个超参数值的预期提升。
- 采样阶段（Sampling Phase）：在这个阶段，TPE 算法根据建模阶段计算得到的指标，从先验概率密度中选择一组新的超参数值进行尝试。具体来说，TPE 算法会根据计算得到的指标，在先验概率密度中使用抽样方法（如抽样方法或概率密度函数的最大化）来选择下一个尝试的超参数值。然后，根据选择的超参数值进行实验，计算目标函数的表现，并更新建模阶段中的先验概率密度和条件概率密度。

通过不断地迭代建模阶段和采样阶段，TPE 算法能够有效地在超参数空间中搜索到最优的超参数组合，从而优化目标函数的性能。

---

**对比 Spark DataFrame 和 Pandas DataFrame**

Spark DataFrame 和 Pandas DataFrame 是两种不同的数据结构，分别用于处理大数据和小数据。

1. **分布式处理**：
   - Spark DataFrame 是 Apache Spark 提供的分布式数据处理框架中的一部分，用于处理大规模数据集。它可以在集群上并行处理数据，具有高度的扩展性和并行性。
   - Pandas DataFrame 是 Python 中的一个库，用于处理较小规模的数据。它是在单个机器上运行的，通常用于处理可以放入内存的数据集。

2. **数据规模**：
   - Spark DataFrame 可以处理超出单个节点内存容量的数据集，适用于大规模数据集的处理和分析。
   - Pandas DataFrame 适用于较小规模的数据集，通常处理的数据量不超过单个节点内存的容量。

3. **操作方式**：
   - Spark DataFrame 支持的操作和转换通常是惰性执行的，需要通过触发动作（如 count()、show() 等）来触发执行。这样可以优化执行计划并最大化并行执行。
   - Pandas DataFrame 是即时执行的，每个操作都会立即执行，并且会将整个数据集加载到内存中。

4. **语言支持**：
   - Spark DataFrame 可以通过多种编程语言访问，包括 Scala、Java、Python 和 R 等。
   - Pandas DataFrame 是 Python 的一个库，主要用于 Python 编程语言。

5. **生态系统**：
   - Spark DataFrame 通常与 Spark 生态系统中的其他组件（如 Spark SQL、MLlib、GraphX 等）集成，可以进行更复杂的数据处理和分析。
   - Pandas DataFrame 可以与 Python 的其他库（如 NumPy、Matplotlib 等）无缝集成，提供丰富的数据分析和可视化功能。

总的来说，Spark DataFrame 适用于处理大规模的分布式数据集，而 Pandas DataFrame 则适用于处理较小规模的单机数据集。选择使用哪种 DataFrame 取决于数据规模、数据处理需求和可用资源等因素。

---

**SME**

在进行数据处理的时候，有时候要看SME的意见。

SME 是指 Subject Matter Expert，即专业领域专家。在各种领域的工作中，SME 是指对特定领域具有深入了解和专业知识的人员。这些人通常在其领域内有着丰富的经验和专业技能，并且对该领域的相关问题、流程、技术、最佳实践等有着深刻的理解。

SME 可以在各种领域中发挥重要作用，例如：

1. 在产品开发中，SME 可以提供关于产品需求、功能设计、技术实现等方面的专业意见和建议。
2. 在项目管理中，SME 可以就项目范围、业务流程、风险评估等方面提供专业指导和支持。
3. 在培训和教育领域，SME 可以担任教师、讲师或培训师，传授自己的专业知识和经验。
4. 在咨询和咨询服务领域，SME 可以提供专业咨询，帮助客户解决各种复杂的问题和挑战。

总之，SME 是指对特定领域具有深入了解和专业知识的专家，在各种领域的工作和项目中起着至关重要的作用。他们的专业知识和经验可以帮助组织和团队更好地理解和解决各种问题，推动项目的顺利实施和业务的持续发展。

---

**算法ALS**

ALS（Alternating Least Squares，交替最小二乘法）是一种常用于推荐系统中的协同过滤算法。协同过滤是一种推荐系统技术，通过分析用户与项目之间的历史交互数据（如用户评分、点击、购买等），来预测用户对尚未交互的项目的喜好程度。

ALS 算法的核心思想是通过交替优化两个矩阵来求解用户和项目之间的潜在关系。具体来说，ALS 算法将用户-项目交互矩阵分解为两个低秩矩阵：用户因子矩阵和项目因子矩阵。用户因子矩阵表示用户对潜在特征的偏好，而项目因子矩阵表示项目在这些潜在特征上的表现。

ALS 算法的求解过程分为两个步骤，交替进行：

1. **固定用户因子矩阵，优化项目因子矩阵**：在这一步中，固定用户因子矩阵，通过最小化损失函数（通常是均方误差）来优化项目因子矩阵，使得预测评分与实际评分尽可能接近。

2. **固定项目因子矩阵，优化用户因子矩阵**：在这一步中，固定项目因子矩阵，通过最小化损失函数来优化用户因子矩阵，同样使得预测评分与实际评分尽可能接近。

通过交替优化这两个矩阵，直到达到一定的迭代次数或收敛条件，从而得到用户因子矩阵和项目因子矩阵。通过这两个矩阵的乘积，可以预测用户对未评分项目的评分，从而进行推荐。

ALS 算法的优点是计算简单、容易实现，并且在处理大规模数据集时具有良好的扩展性。因此，它在推荐系统中得到了广泛的应用。

我遇到这个算法是在 Notebook 中的，对**缺失值补全的**技术。一种技术，有很多用法。

---
**Scala 和 Spark**

Scala 和 Apache Spark 之间有着密切的关系，Scala 是 Spark 的首选编程语言之一，它们之间存在着以下几种关系：

1. **编程语言**：
   - Scala 是一种多范式编程语言，结合了面向对象编程和函数式编程的特性。而 Apache Spark 是用 Scala 编写的分布式计算引擎，Scala 是 Spark 的主要编程语言之一。
   - 在 Spark 的早期版本中，Scala 是 Spark 的默认编程语言，因为 Spark 本身就是用 Scala 编写的。因此，Scala 成为了 Spark 社区中最为流行的编程语言之一。

2. **API 设计**：
   - Spark 的 Scala API 是最早和最完善的 API 之一，因此许多 Spark 用户和开发者首选使用 Scala 来编写 Spark 应用程序。Scala API 提供了对 Spark 强大功能的完整访问，并且性能优越，因为 Scala 与 Spark 的内部实现紧密集成。

3. **交互式开发**：
   - Scala 也是 Spark Shell 的默认交互式编程语言之一，开发者可以在 Scala Shell 中直接执行 Spark 代码，并实时查看结果，快速验证和调试代码。

4. **生态系统**：
   - Scala 作为一种 JVM 语言，与 Java 和其他 JVM 语言的互操作性非常好。因此，许多 Java 开发者也可以轻松地使用 Scala 来编写 Spark 应用程序，从而丰富了 Spark 的生态系统。
   - 除了 Scala 之外，Spark 也提供了 Python、Java 和 R 等多种编程语言的 API，以满足不同开发者的需求。

综上所述，Scala 是 Apache Spark 的首选编程语言之一，它与 Spark 之间有着紧密的关系，为 Spark 应用程序的开发和使用提供了强大的支持和便利。

---

决策树算法中不推荐对数据进行 One-Hot 编码的原因主要有以下几点：

1. **特征表达能力：** 决策树算法本身能够很好地处理分类特征。它可以通过在节点处根据特征值进行划分来处理分类特征，因此不需要对分类特征进行 One-Hot 编码。在决策树算法中，直接使用分类特征的原始值可以更好地保留特征的信息。

2. **编码引入的维度灾难：** One-Hot 编码会将一个有限的分类特征转换为多个二元特征（虚拟变量），从而引入了大量的额外维度，特别是在原始特征具有多个类别时。这样会增加特征空间的维度，导致模型复杂度增加，增加了计算的开销，同时可能会引起过拟合问题。

3. **稀疏性：** 在数据集中有大量的分类特征时，One-Hot 编码会生成大量的零值特征，导致数据变得非常稀疏。这样会增加模型的内存消耗和计算时间，并且可能会降低模型的性能。

4. **树的分裂策略：** 决策树算法通常使用启发式方法选择最佳分裂点，而这些方法可以直接处理分类特征的原始值。因此，不需要对分类特征进行 One-Hot 编码。

总的来说，在决策树算法中，直接使用分类特征的原始值而不进行 One-Hot 编码可以保持模型的简洁性和可解释性，并且可以避免引入额外的维度和稀疏性问题。因此，不推荐对数据进行 One-Hot 编码。

---
**决策树的 Scale Invariant**

在决策树算法中，"scale invariant" 意味着决策树对特征的尺度变换是不敏感的。换句话说，如果对数据集中的某个特征进行线性尺度变换（如缩放、归一化），决策树模型的输出不会受到影响。

这种性质对于决策树是非常重要的，因为决策树的分裂过程是基于特征的相对顺序而不是绝对值。在决策树的分裂过程中，每个节点都会选择一个特征和一个阈值来进行分裂，而这个选择通常是基于特征的排序而不是其绝对值。因此，如果对特征进行线性尺度变换，不会改变特征之间的相对顺序，决策树的分裂结果也会保持不变。

举个例子，假设有一个特征表示身高，取值范围为 150cm 到 200cm。如果将这个特征进行缩放，变换到 1.5 到 2.0 之间，决策树在选择分裂点时不会因为特征的缩放而改变，因为它仍然会根据特征的相对顺序来进行选择。

因此，决策树的"scale invariant"性质使得它在处理不同尺度的特征时更加灵活和鲁棒，无需对特征进行额外的缩放或归一化处理。

---
**决策树的Pitfall**

决策树在使用时可能会遇到一些潜在的问题（pitfall），这些问题可能会影响模型的性能和泛化能力。以下是一些常见的决策树可能遇到的 pitfalls：

1. **过拟合（Overfitting）：** 决策树容易过拟合训练数据，特别是在树的深度很大的情况下。过拟合会导致模型在训练集上表现很好，但在未见过的数据上表现很差。

2. **高方差（High Variance）：** 决策树的高方差意味着对训练数据中的噪声敏感，导致模型在不同的训练集上产生不稳定的预测结果。

3. **局部最优解（Local Optima）：** 决策树是一种贪婪算法，它在每个节点上选择当前最优的分裂点，但这种局部最优的选择不一定能够保证全局最优的结果。

4. **特征选择偏差（Feature Selection Bias）：** 决策树倾向于选择高度相关的特征进行分裂，而忽略其他可能对预测有用的特征。这可能导致模型忽略了重要的特征或者选择了不够优秀的特征。

5. **不稳定性（Instability）：** 决策树对输入数据的微小变化可能会产生较大的变化，导致模型的不稳定性。这使得模型难以在不同的数据集上泛化。

6. **处理连续型特征的不足（Handling Continuous Features）：** 决策树处理连续型特征时通常使用离散化（binning）的方法，这可能会导致信息损失和模型的不准确性。

7. **处理类别不平衡的数据（Handling Class Imbalance）：** 当类别不平衡时，决策树可能会偏向于预测出现频率较高的类别，而忽略罕见类别。

为了解决这些问题，可以采取一些方法，如剪枝（Pruning）、集成学习（Ensemble Learning，如随机森林）、调整超参数、特征选择、数据增强等。这些方法有助于提高决策树模型的泛化能力和性能。

# 5 - 参考内容

[官方教程](https://spark.apache.org/docs/latest/api/python/index.html)和[documentation](https://spark.apache.org/docs/latest)是最好的学习资料。

[Documentation](https://docs.databricks.com/en/machine-learning/index.html)

[Full Code](https://github.com/sherryuuer/machine-learning-lab/tree/main/Databricks)
