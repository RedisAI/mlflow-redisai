from pathlib import Path
import sys

import torch
from mlflow.models import Model


FLAVOR_NAME = "torchscript"
_SERIALIZED_TORCH_MODEL_FILE_NAME = "model.pt"


def save_model(torchscript_model, path, mlflow_model=Model(), **kwargs):
    """
       Save a PyTorch model (torchscripted) to a path on the local file system. TorchScript
       models can be made using scripting or tracing. For more details check out the
       official documentation https://pytorch.org/docs/stable/jit.html.

       :param torchscript_model: JITed PyTorch model to be saved.
       :param path: Local path where the model is to be saved.
       :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
       :param kwargs: kwargs to pass to ``torch.jit.save`` method.

       >>> import torch
       >>> import mlflow
       >>> import mlflow.pytorch
       >>> # create model and set values
       >>> pytorch_model = Model()
       >>> pytorch_model_path = ...
       >>> #train our model
       >>> for epoch in range(500):
       >>>     y_pred = model(x_data)
       >>>     ...
       >>> #save the model
       >>> with mlflow.start_run() as run:
       >>>   mlflow.log_param("epochs", 500)
       >>>   torchscript_model = torch.jit.script(pytorch_model)
       >>>   mlflow.pytorch.save_model(torchscript_model, pytorch_model_path)
       """
    if not isinstance(torchscript_model, torch.jit.ScriptModule):
        raise TypeError("Argument 'torchscript_model' should be a torch.jit.ScriptModule")

    path = Path(path)
    if path.exists():
        raise RuntimeError("Path '{}' already exists".format(path))
    path.mkdir()

    model_data_subpath = "data"
    model_data_path = path/model_data_subpath
    model_data_path.mkdir()

    model_path = model_data_path/_SERIALIZED_TORCH_MODEL_FILE_NAME
    torchscript_model.save(str(model_path))

    mlflow_model.add_flavor(
        FLAVOR_NAME, model_data=model_data_subpath, pytorch_version=torch.__version__)

    mlflow_model.save(path/"MLmodel")


def log_model(torchscript_model, artifact_path, registered_model_name=None, **kwargs):
    """
    Log a PyTorch model (TorchScripted) as an MLflow artifact for the current run.
    :param torchscript_model: : JITed PyTorch model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param registered_model_name: Note:: Experimental: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param kwargs: kwargs to pass to ``torch.save`` method.

       >>> import torch
       >>> import mlflow
       >>> import mlflow.pytorch
       >>> # create model and set values
       >>> pytorch_model = Model()
       >>> pytorch_model_path = ...
       >>> #train our model
       >>> for epoch in range(500):
       >>>     y_pred = model(x_data)
       >>>     ...
       >>> #save the model
       >>> with mlflow.start_run() as run:
       >>>   mlflow.log_param("epochs", 500)
       >>>   torchscript_model = torch.jit.script(pytorch_model)
       >>>   mlflow.pytorch.log_model(torchscript_model, pytorch_model_path)
    """

    # TODO: use registered_model_name
    Model.log(artifact_path=artifact_path, flavor=sys.modules[__name__],
              torchscript_model=torchscript_model, conda_env=None, code_paths=None,
              pickle_module=None, registered_model_name=registered_model_name, **kwargs)

