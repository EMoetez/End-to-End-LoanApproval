import pyspark
from pyspark.sql import SparkSession
#from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyparsing import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.feature import StandardScalerModel
#from  model.process import DataProcessor
import os



path="Model_deployment/"
sample_processed_directory = "Model Development/sample/processed"
sample_raw_directory = "Model Development/sample/raw"

# Initialize Spark session
spark = SparkSession.builder.appName("CreditCardPrediction").getOrCreate()

# #testing the current directory
print(os.path.exists(path))  # Should print True if the path exists
print(os.getcwd())


class DataProcessor:
    def __init__(self, scaler_model_path="Model Development/scaler", feature_columns=None):
        """
        Initialize the DataProcessor class.
        :param scaler_model_path: Path to the pre-trained scaler model.
        :param feature_columns: List of feature column names to be used for modeling.
        """
        self.feature_columns = feature_columns if feature_columns is not None else [
            'INCOME_PER_FAM_MEMBER', 'AMT_INCOME_TOTAL', 'YEARS_EMPLOYED', 'AGE', 'CREDIT_HISTORY_LENGTH', 'RECENT_ACTIVITY'
        ]
        self.scaler_model = StandardScalerModel.load(scaler_model_path)

    def process(self, input_df, output_dir):
        """
        This method processes the input data, applies all transformations, 
        and saves the output to the specified directory.
        :param input_df: Spark DataFrame to be processed.
        :param output_dir: Directory to save the processed data.
        :param file_format: File format to save the data (default is "parquet").
        """
        # Ordinal Encoding for specific categorical columns
        categorical_ordinal_columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        
        # Apply StringIndexer for ordinal columns
        indexers = [StringIndexer(inputCol=col, outputCol=col + "_encoded") for col in categorical_ordinal_columns]
        pipeline = Pipeline(stages=indexers)
        indexed_df = pipeline.fit(input_df).transform(input_df)

        # OneHotEncoding for specific categorical columns
        categorical_onehot_columns = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
        
        # Apply StringIndexer for one-hot encoding
        indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_onehot_columns]
        indexer_pipeline = Pipeline(stages=indexers)
        indexed_onehot_df = indexer_pipeline.fit(input_df).transform(input_df)

        # Check for distinct values before applying OneHotEncoder
        valid_columns = []
        for col in categorical_onehot_columns:
            distinct_values = indexed_onehot_df.select(col + "_index").distinct().count()
            if distinct_values >= 2:
                valid_columns.append(col)

        # Apply OneHotEncoder only to valid columns
        encoder = OneHotEncoder(inputCols=[col + "_index" for col in valid_columns],
                                outputCols=[col + "_onehot" for col in valid_columns])
        encoded_df = encoder.fit(indexed_onehot_df).transform(indexed_onehot_df)
        
        # Custom feature engineering
        input_df = input_df.withColumn("AGE", F.abs(input_df['DAYS_BIRTH']) / 365)
        input_df = input_df.withColumn("YEARS_EMPLOYED", F.abs(input_df['DAYS_EMPLOYED']) / 365)

        # Drop unnecessary columns
        input_df = input_df.drop('DAYS_BIRTH', 'DAYS_EMPLOYED')

        # Create new features (e.g., 'INCOME_PER_FAM_MEMBER')
        input_df = input_df.withColumn("INCOME_PER_FAM_MEMBER", input_df['AMT_INCOME_TOTAL'] / input_df['CNT_FAM_MEMBERS'])

        # Feature engineering for 'CREDIT_HISTORY_LENGTH' and 'RECENT_ACTIVITY'
        # Convert 'MONTHS_BALANCE' to numeric type (integer)
        input_df = input_df.withColumn("MONTHS_BALANCE", F.col("MONTHS_BALANCE").cast("int"))

        # Now perform the subtraction operation
        credit_history_length = input_df.groupBy('ID').agg(
            (F.max('MONTHS_BALANCE') - F.min('MONTHS_BALANCE')).alias('CREDIT_HISTORY_LENGTH')
        )
        recent_activity_flag = input_df.groupBy('ID').agg(
            (F.expr("max(CASE WHEN MONTHS_BALANCE >= -4 THEN 1 ELSE 0 END)")).alias('RECENT_ACTIVITY')
        )

        # Join these features back into the main dataframe
        input_df = input_df.join(credit_history_length, on="ID", how="left")
        input_df = input_df.join(recent_activity_flag, on="ID", how="left")

        # Select features for modeling
        selected_features = ['INCOME_PER_FAM_MEMBER', 'AMT_INCOME_TOTAL', 'YEARS_EMPLOYED', 'AGE', 'CREDIT_HISTORY_LENGTH', 'RECENT_ACTIVITY']

        # Create a VectorAssembler to combine features into a single vector column
        assembler = VectorAssembler(inputCols=selected_features, outputCol="features")

        # Apply VectorAssembler to prepare features
        feature_df = assembler.transform(input_df)

        # Apply the pre-trained scaler model
        train_data_scaled = self.scaler_model.transform(feature_df)

        # Save the processed data
        train_data_scaled.write.parquet(output_dir, mode='overwrite')
        return train_data_scaled




# Load the trained model from Parquet
def load_model(model_path="Model_Deployment/rf_model"):
    """
    Load the trained model stored in Parquet format.
    """
    rf_model = RandomForestClassificationModel.load(model_path)
    print("RandomForest model loaded successfully!")
    return rf_model

 


# Process data using DataProcessor class
def preprocess_data(raw_directory= sample_raw_directory,processed_directory= sample_processed_directory):
    
    sample_data = spark.read.parquet(raw_directory)
    data_processor = DataProcessor()
    sample_processed = data_processor.process(input_df=sample_data,output_dir=processed_directory)
    #sample_processed.show()
    return sample_processed

# predict from a given directory, require a parquet format for both data and model
def predict(processed_directory: str, model:pyspark.ml.classification.RandomForestClassificationModel) -> dict:
    """
    Predict using the loaded model.
    :param features: Input features for prediction.
    :param model: The loaded model.
    :return: Prediction result as a dictionary.
    """
    test_data_loaded = spark.read.parquet(processed_directory)

    # Convert input features to a Spark DataFrame
    #feature_df = spark.createDataFrame([features])
    
    # Apply the model
    prediction = model.transform(test_data_loaded)
    prediction_result = prediction.select("prediction").collect()[0][0]
    
    return {"prediction": prediction_result}

