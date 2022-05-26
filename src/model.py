import time
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import mlflow
import config
warnings.filterwarnings('ignore')

TRACKING_URL = config.TRACKING_URL


class Model:
    """
    save and train functions for models
    """
    def __init__(self,df=None) -> None:
        self.df = df
        self.client = MlflowClient(registry_uri=TRACKING_URL)

    def wait_model_transition(self, model_name, model_version, stage):
        """wait_model_transition"""
        client = self.client
        for _ in range(10):
            model_version_details = client.get_model_version(name=model_name,
                                                             version=model_version,
                                                             )
            status = ModelVersionStatus.from_string(model_version_details.status)
            print(f"Model status: {ModelVersionStatus.to_string(status)}")
            if status == ModelVersionStatus.READY:
                client.transition_model_version_stage(
                  name=model_name,
                  version=model_version,
                  stage=stage,
                )
                break
            time.sleep(1)

    def save_model(self,artifact_path, model, experiment_name, accuracy_train,accuracy_test,roc_auc):
        """
        save model using mlflow.
        database at src/mlflow.db
        mlflowids at src/mlflow-artifcat-root
        """
        #connect to client and set experiment.
        #workarround
        mlflow.set_tracking_uri(TRACKING_URL)
        try:
            mlflow.create_experiment(experiment_name)
            print(mlflow.list_experiments())
        except Exception as e:
            print("-----------------")
            print(e)
            print("-----------------")
        client = self.client
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            run_num = run.info.run_id
            model_uri = f"runs:/{run_num}/{artifact_path}"
            mlflow.log_metric('accuracy_train', accuracy_train)
            mlflow.log_metric('accuracy_test', accuracy_test)
            mlflow.sklearn.log_model(model, artifact_path)
            mlflow.register_model(model_uri=model_uri,
                                      name=artifact_path)
        model_version_infos = client.search_model_versions(f"name = '{artifact_path}'")
        new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
        # update model version with latest version
        client.update_model_version(
        name=artifact_path,
        version=new_model_version,
        description=experiment_name
        )
        # Necessary to wait to version models
        try:
            # Move the previous model to None version
            self.wait_model_transition(artifact_path, int(new_model_version)-1, "None")
        except:
            # Move the latest model to Staging (could also be Production)
            self.wait_model_transition(artifact_path, new_model_version, "Staging")


    def random_forest(self,t_columns, symbol):
        """ random_forest """
        df_v = self.df[t_columns]
        df_l = self.df[f'{symbol}_Shifted_Log_Return']
        # df_v=df_v.iloc[: , :-2]
        n_test = int(len(df_l)*80/100)
        train_msft = df_v.iloc[1:n_test]
        test_msft = df_v.iloc[n_test:-1]
        train_msft_l = df_l.iloc[1:n_test]
        test_msft_l = df_l.iloc[n_test:-1]
        c_train = (train_msft_l>0)
        c_test = (test_msft_l>0)
        params= {'max_depth': 2048,
        'max_features': 5,
        'min_samples_leaf': 4,
        'min_samples_split': 4,
        'n_estimators': 800}
        model = RandomForestClassifier(**params)
        model.fit(train_msft, c_train)
        y_test_pp = model.predict_proba(test_msft)
        accuracy_train = model.score(train_msft, c_train)
        accuracy_test = model.score(test_msft, c_test)
        roc_auc = roc_auc_score(c_test, y_test_pp[:,1])
        symbol += 'r_f'
        fpr, tpr, thres = roc_curve(c_test, y_test_pp[:,1])
        self.save_model(symbol,model,'random_f',accuracy_train,accuracy_test,roc_auc)
        return [accuracy_train,accuracy_test,roc_auc,fpr,tpr,thres]

    def logistic_reg(self, t_columns, symbol):
        """ random_forest """
        df_v = self.df[t_columns]
        df_l = self.df[f'{symbol}_Shifted_Log_Return']
        # df_v=df_v.iloc[: , :-2]
        n_test = int(len(df_l)*80/100)
        train_msft = df_v.iloc[1:n_test]
        test_msft = df_v.iloc[n_test:-1]
        train_msft_l = df_l.iloc[1:n_test]
        test_msft_l = df_l.iloc[n_test:-1]
        c_train = (train_msft_l>0)
        c_test = (test_msft_l>0)
        model = LogisticRegression(C=10)
        model.fit(train_msft, c_train)
        y_test_pp = model.predict_proba(test_msft)
        accuracy_train = model.score(train_msft, c_train)
        accuracy_test = model.score(test_msft, c_test)
        roc_auc = roc_auc_score(c_test, y_test_pp[:,1])
        symbol += 'l_r'
        fpr, tpr, thres = roc_curve(c_test, y_test_pp[:,1])
        self.save_model(symbol,model,'logistic_f',accuracy_train,accuracy_test,roc_auc)
        return [accuracy_train,accuracy_test,roc_auc,fpr,tpr,thres]
