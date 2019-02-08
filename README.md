# pyspark-ML-in-Colab
Pyspark in Google Colab: A simple machine learning (Linear Regression) model

Pyspark in Google Colab: A simple machine learning (Linear Regression) model


Why do we need Spark?


With broadening sources of the data pool, the topic of Big Data has received an increasing amount of attention in the past few years. Besides dealing with the gigantic data of all kinds and shapes, the target turnaround time of the analysis part for the big data has been reduced significantly. The speed and efficiency have made possible for an immediate analysis of the Big Data and use it to identify new opportunities. This, in turn, leads to smarter business moves, more efficient operations, higher profits, and happier customers.




Yes, but why Google Colab?


Colab by Google is based on Jupyter Notebook which is an incredibly powerful tool that leverages google docs features. Since it runs on google server, we don't need to install anything in our system, be it Spark or deep learning model. Probably the most attractive features of Colab is the free GPU and TPU support! Since it is run on Google's own server, the GPU support, in fact, faster than some commercially available GPUs for eg. Nvidia 1050Ti. A piece of general system information allocated for a user looks like the following:
Gen RAM Free: 11.6 GB  | Proc size: 666.0 MB
GPU RAM Free: 11439MB | Used: 0MB | Util  0% | Total 11439MB
If you are interested to know more about Colab, this article by Anna Bonner points out some of the outstanding benefits.
Enough of the small talks. Let's create a simple linear regression model with PySpark in Google Colab. 
To open your first Colab Jupyter Notebook, click on this link.


Running Pyspark in Colab


To run spark in Colab, we need to first install all the dependencies in Colab environment i.e. Apache Spark 2.3.2 with hadoop 2.7, Java 8 and Findspark to locate the spark in the system. The tools installation can be carried out inside the Jupyter Notebook of the Colab. One important note is that if you are new in Spark, it is better to avoid Spark 2.4.0 version since some people have already complained about its compatibility issue with python. 
Follow the steps to install the dependencies:
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://apache.osuosl.org/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz
!tar xf spark-2.3.2-bin-hadoop2.7.tgz
!pip install -q findspark
 Now that you installed Spark and Java in Colab, it is time to set the environment path which enables you to run Pyspark in your Colab environment. Set the location of Java and Spark by running the following code:
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.3.2-bin-hadoop2.7"
Run a local spark session to test your installation:
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
Congrats! Your Colab is ready to run Pyspark. Let's build a simple Linear Regression model.


Linear Regression Model


Linear Regression model is one the oldest and widely used machine learning approach which assumes a relationship between dependent and independent variables. For example, a modeler might want to predict the forecast of the rain based on the humidity ratio. Linear Regression consists of the best fitting line through the scattered points on the graph and the best fitting line is known as the regression line. Detailed about linear regression can be found here.
For our purpose of starting with Pyspark in Colab and to keep things simple, we will use the famous Boston Housing dataset. A full description of this dataset can be found in this link. 
The goal of this exercise to predict the housing prices by the given features. Let's predict the prices of the Boston Housing dataset by considering MEDV as the output variable and all the other variables as input.
Download the dataset from here and keep it somewhere on your computer. Load the dataset into your Colab directory from your local system:
from google.colab import files
files.upload()
Check if it is uploaded correctly in the system by the following command
!ls
You should see a file named BostonHousing.csv. Now that we have uploaded the dataset, we can start analyzing. 
For our linear regression model to run we need to import two modules from Pyspark i.e. Vector Assembler and Linear Regression. Vector Assembler is a transformer that assembles all the features into one vector from multiple columns that contain type double. We could have used StringIndexer if any of our columns contains string values to convert it into numeric values. Luckily, the BostonHousing dataset only contains double values, so we don't need to worry about StringIndexer for now.
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
dataset = spark.read.csv('BostonHousing.csv',inferSchema=True, header =True)
Notice that we used InferSchema inside read.csv mofule. InferSchema enables us to infer automatically different data types for each column.
Let us print look into the dataset to see the data types of each column:
dataset.printSchema()
It should print the data types as follows:
Next step is to convert all the features from different columns into a single column and let's call this new vector column as 'Attributes' in the outputCol. 
#Input all the features in one vector column
assembler = VectorAssembler(inputCols=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'], outputCol = 'Attributes')
output = assembler.transform(dataset)
#Input vs Output
finalized_data = output.select("Attributes","medv")
finalized_data.show()
The output is:
Here, 'Attributes' are in the input features from all the columns and 'medv' is the target column.
Next, we should split the training and testing data according to our dataset (0.8 and 0.2 in this case).  
#Split training and testing data
train_data,test_data = finalized_data.randomSplit([0.8,0.2])
regressor = LinearRegression(featuresCol = 'Attributes', labelCol = 'medv')
#Learn to fit the model from training set
regressor = regressor.fit(train_data)
#To predict the prices on testing set
pred = regressor.evaluate(test_data)
#Predict the model
pred.predictions.show()
The predicted score in the prediction column as output:


We can also print the coefficient and intercept of the regression model by using the following command:
#coefficient of the regression model
coeff = regressor.coefficients
#X and Y intercept
intr = regressor.intercept
print ("The coefficient of the model is : %a" %coeff)
print ("The Intercept of the model is : %f" %intr)
Once we are done with the basic linear regression operation, we can go a bit further and analyze our model statistically by importing RegressionEvaluator module from Pyspark. 
from pyspark.ml.evaluation import RegressionEvaluator
eval = RegressionEvaluator(labelCol="medv", predictionCol="prediction", metricName="rmse")
# Root Mean Square Error
rmse = eval.evaluate(pred.predictions)
print("RMSE: %.3f" % rmse)
# Mean Square Error
mse = eval.evaluate(pred.predictions, {eval.metricName: "mse"})
print("MSE: %.3f" % mse)
# Mean Absolute Error
mae = eval.evaluate(pred.predictions, {eval.metricName: "mae"})
print("MAE: %.3f" % mae)
# r2 - coefficient of determination
r2 = eval.evaluate(pred.predictions, {eval.metricName: "r2"})
print("r2: %.3f" %r2)
That's it. Your first machine learning using Pyspark in Google Colab.
