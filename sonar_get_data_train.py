import os

from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath
from kfp import kubernetes


@dsl.component(base_image="quay.io/arckrish/custom-wb-images:jupyter-pytorch-ubi8-python-3.9-vz1.0_20240711")
def get_data(data_output_path: OutputPath()):
    import urllib.request
    print("starting download...")
    url = "https://github.com/arckrish/Train-Sample-Data.git/main/data/sonar.csv"
    urllib.request.urlretrieve(url, data_output_path)
    print("done")

@dsl.component(
    base_image="quay.io/arckrish/custom-wb-images:jupyter-pytorch-ubi8-python-3.9-vz1.0_20240711",
    packages_to_install=["tf2onnx", "seaborn", "pandas", "torch", "torch.nn", "torch.optim"],
)
def train_model(data_input_path: InputPath(), model_output_path: OutputPath()):
    import numpy as np
    import pandas as pd
    import torch
    import tf2onnx
    import onnx
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Read data, convert to NumPy arrays
    data = pd.read_csv("data/sonar.csv", header=None)
    X = data.iloc[:, 0:60].values
    y = data.iloc[:, 60].values
    data.head()

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # convert into PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # train-test split for evaluation of the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # set up DataLoader for training set
    loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)


    # create model
    model = nn.Sequential(
        nn.Linear(60, 60),
        nn.ReLU(),
        nn.Linear(60, 30),
        nn.ReLU(),
        nn.Linear(30, 1),
        nn.Sigmoid()
    )


    # Train the model
    n_epochs = 200
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save the model as ONNX for easy use of ModelMesh

    model_proto, _ = tf2onnx.convert.from_keras(model)
    print(model_output_path)
    onnx.save(model_proto, model_output_path)


@dsl.component(
    base_image="quay.io/arckrish/custom-wb-images:jupyter-pytorch-ubi8-python-3.9-vz1.0_20240711",
    packages_to_install=["boto3", "botocore"]
)

def upload_model(input_model_path: InputPath()):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    s3_key = os.environ.get("S3_KEY")

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading {s3_key}")
    bucket.upload_file(input_model_path, s3_key)


@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline():
    get_data_task = get_data()
    csv_file = get_data_task.outputs["data_output_path"]
    # csv_file = get_data_task.output
    train_model_task = train_model(data_input_path=csv_file)
    onnx_file = train_model_task.outputs["model_output_path"]

    upload_model_task = upload_model(input_model_path=onnx_file)

    upload_model_task.set_env_variable(name="S3_KEY", value="models/sonar/1/model.onnx")

    kubernetes.use_secret_as_env(
        task=upload_model_task,
        secret_name='aws-connection-my-storage',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )
