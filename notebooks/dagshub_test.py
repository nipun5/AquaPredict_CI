import dagshub
dagshub.init(repo_owner='nipun5', repo_name='AquaPredict_CI', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)