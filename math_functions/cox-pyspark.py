from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import exp

spark = SparkSession.builder.appName("DistributedCoxRegression").getOrCreate()

df = spark.read.csv("data.csv", header=True, inferSchema=True)

assembler = VectorAssembler(inputCols=["features"], outputCol="features_vec")
df = assembler.transform(df)

def cox_partial_log_likelihood(features, time, event, coefficients):
    hazard = exp(features.dot(coefficients))
    risk = sum(hazard)
    partial_ll = event * (features - features.multiply(hazard / risk))
    return partial_ll

def cox_regression(df, label_col, features_col, censor_col, max_iter=100, lr=0.1, tol=1e-6):
    beta = Vectors.dense([0.0] * len(df.select(features_col).first()[features_col]))
    for i in range(max_iter):
        prev_beta = beta.copy()
        partial_log_likelihood_udf = udf(lambda features, time, event: cox_partial_log_likelihood(features, time, event, prev_beta), VectorUDT())
        partial_ll = df.withColumn("partial_ll", partial_log_likelihood_udf(col(features_col), col(label_col), col(censor_col)))
        hazard_ratio_udf = udf(lambda features: exp(features.dot(prev_beta)), DoubleType())
        hazard_ratio = partial_ll.select(hazard_ratio_udf(col(features_col)).alias("hazard_ratio")).agg({"hazard_ratio": "sum"}).collect()[0]["sum(hazard_ratio)"]
        grad = partial_ll.selectExpr("sum(partial_ll) as partial_ll").rdd.map(lambda row: row.partial_ll).reduce(lambda a, b: a + b).toArray()
        beta += lr * grad
        if Vectors.dense(np.abs(beta - prev_beta)).max() < tol:
            break
    return beta, hazard_ratio

beta, hazard_ratio = cox_regression(df, "time", "features_vec", "event")

