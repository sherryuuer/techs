## Databricks ML

---
### 1 - Delta Lake 概念

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

### 2 - 在Azure的workspace环境中运行笔记本

创建一个分布式的cluster环境后就可以打开笔记本了，整个环境是在一个workspace中的，虽然自己对Azure接触的不多，但是从公司提供的环境看，接入环境，启动环境集成服务器，以及进入笔记本都非常方便。

但是每次更换笔记本的时候都要重新手动接上cluster，这个要注意。

**Delta数据类型**

- 可以将数据读入 delta table 进行使用。
- 可以对 delta table 进行读写更新操作。
- 可以进行时间旅行，回溯到上一个版本。
- 支持所有的 Pyspark 语法

**数据清洗 Data Cleansing**

- `df.describe()` 和 `df.summary()` 是对spark dataframe的统计描述，`summary()` 比 `describe()` 增加了四分位数的描述。
- `dbutils.data.summarize(df)` 可以对数据进行更详细的统计分析，最后一列是一个可视化图表，非常耳目一新。
- 在 Spark ML（Spark机器学习）中，数值通常被表示为 double 类型。这主要是因为 double 类型在存储精度和数值范围上都比较适合机器学习中的数据表示和计算。所以在数据清洗的时候要将整数 `integerType()` 进行 `.cast('double')` 的处理。
- 缺失值标记：

```python
for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))
```

- Transformers 是一类用于将数据集进行转换的对象。它们接受一个 DataFrame 作为输入，并生成一个新的 DataFrame 作为输出。常见的转换操作包括特征处理（如特征提取、特征转换、特征选择等）、数据清洗、数据规范化等。主要通过 `transform()` 方法来进行转换操作。（注意，它的变换不是基于学习，而是基于规则。）
- Estimators 是一类用于训练模型的对象。它们接受一个 DataFrame 作为输入，并返回一个模型（Model）对象。通常，Estimators 是通过对输入数据进行学习（即训练）来生成模型的。Estimators 主要通过 `fit()` 方法来进行训练操作。
- 在 Spark MLlib 中，Transformers 和 Estimators 通常被组合使用，构建成一个数据处理和建模的流水线（Pipeline）。这种流水线的设计使得用户可以将数据处理和建模过程整合在一起，并且可以很方便地将不同的数据处理步骤和建模步骤组合起来，形成一个完整的数据处理和建模流程。

```python
from pyspark.ml.feature import Imputer
imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)
imputer_model = imputer.fit(doubles_df)
imputed_df = imputer_model.transform(doubles_df)
# 将清洗好的数据存入 delte，就可以进行下一步的模型训练
imputed_df.write.format("delta").mode("overwrite").save(f"{DA.paths.working_dir}/imputed_results")
```

**线性回归 Linear Regression**

- 独特的数据分割方法，对训练集和测试集进行分区并进行分割。（传统我们用的都是sklearn的 `model_selection` 的 `train_test_split`），下面是分区的代码：
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
- categorical 数据类型的处理：这里需要进行两次转换，从字符串到索引，再从索引到独热编码。

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
学习notebook的代码如下：

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

**MLFlow**

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

**MLflow Model Registry**

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

**Single Decision Trees**

- 决策树算法中不需要对字符串特征进行OHE操作。
- 决策树算法特征重要度：`dt_model.featureImportances`。是对打包前的特征量的重要度一览，但是重要度为0的特征会不显示。使用以下代码 Pandas 将其变成可读的 dataframe。
```python
features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), dt_model.featureImportances)), columns=['feature', 'importance'])
```
- 以上可以得出大部分特征重要度为 0 ，是因为参数 maxDepth 默认设置为5了，只有 5 个特征量被使用了。
- 决策树具有尺度不变性（scale invariant），就算改变数据的尺度，对决策也不会有很大影响。（见概念补充部分）

**Random Forests and Hyperparameter Tuning**

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

**Hyperopt**

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

**AutoML**

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

**Feature Store**

- 是一个存储特征的仓库，字面意思。通过以下代码也可以进行定义。
```python
from databricks import feature_store
fs = feature_store.FeatureStoreClient()
```





### 3 - 知识概念补充

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

---

### 4 - Azure Databricks 的领域内容

- Databricks机器学习 – 29% (13/45)
考察了Databricks独有功能（Cluster、Repos、Workflow、AutoML、Feature Store、MLflow）的规范问题。
关于MLflow的问题出现较多。至少需要理解与Scalable Machine Learning with Apache Spark笔记本中的MLflow相关的代码和UI使用方法。

- ML工作流程 – 29% (13/45)
利用Databricks，考察了ML工作流程中的各个步骤（探索性数据分析、特征工程、调整、模型评估）的方法论问题。
特征工程和调整方面的问题较多，关于特征工程主要涉及到缺失值替换/OneHot编码，关于调整主要涉及到ParamGrid/CrossValidator/Hyperopt等。
关于模型评估，还涉及了常用评估指标的使用场景等通用机器学习问题。

- Spark ML – 33% (15/45)
涉及了Spark ML、Pandas API、Pandas UDF、Pandas Function API等分布式学习机制和API使用方法的问题。
基本上只要掌握了Scalable Machine Learning with Apache Spark中涵盖的内容，就不会有问题。

- 扩展机器学习模型 – 9% (4/45)
考察了决策树算法中maxBins以及矩阵分解等大规模数据机器学习处理的并行化方面的一些较高级的问题。

### 5 - 参考内容

官方的notebook，我自己的github[链接](https://github.com/sherryuuer/machine-learning-lab/tree/main/Databricks-pyspark)，是日文版，官方的已被删除。

阿里巴巴的[文档](https://help.aliyun.com/document_detail/167619.html?spm=a2c4g.167618.0.nextDoc.78563233WGEoGL)也不错。

大象教程[文档](https://www.hadoopdoc.com/spark/spark-principle)挺不错的。
