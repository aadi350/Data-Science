from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
def spark_corr(df, keys=["id"]):
    vector_col = "corr_features"
    col = list(df.drop(*keys).columns)
    assembler = VectorAssembler(
        inputCols=col, outputCol=vector_col, handleInvalid="skip"
    )
    df_vector = assembler.transform(t).select(vector_col)

    matrix = Correlation.corr(df_vector, vector_col)
    cor_np = matrix.collect()[0][matrix.columns[0]].toArray()

    return pd.DataFrame(index=col, columns=col, data=cor_np)
