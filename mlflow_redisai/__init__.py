from pathlib import Path
import logging

import redisai
import ml2rt
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

from . import torchscript
import mlflow.tensorflow


_logger = logging.getLogger(__name__)
SUPPORTED_DEPLOYMENT_FLAVORS = [torchscript.FLAVOR_NAME, mlflow.tensorflow.FLAVOR_NAME]


_flavor2backend = {
    torchscript.FLAVOR_NAME: 'torch',
    mlflow.tensorflow.FLAVOR_NAME: 'tf'}


def _get_preferred_deployment_flavor(model_config):
    """
    Obtains the flavor that MLflow would prefer to use when deploying the model on RedisAI.
    If the model does not contain any supported flavors for deployment, an exception
    will be thrown.

    :param model_config: An MLflow model object
    :return: The name of the preferred deployment flavor for the specified model
    """
    # TODO: add onnx & TFlite
    if torchscript.FLAVOR_NAME in model_config.flavors:
        return torchscript.FLAVOR_NAME
    elif mlflow.tensorflow.FLAVOR_NAME in model_config.flavors:
        return mlflow.tensorflow.FLAVOR_NAME
    else:
        raise MlflowException(
            message=(
                "The specified model does not contain any of the supported flavors for"
                " deployment. The model contains the following flavors: {model_flavors}."
                " Supported flavors: {supported_flavors}".format(
                    model_flavors=model_config.flavors.keys(),
                    supported_flavors=SUPPORTED_DEPLOYMENT_FLAVORS)),
            error_code=RESOURCE_DOES_NOT_EXIST)


def _validate_deployment_flavor(model_config, flavor):
    """
    Checks that the specified flavor is a supported deployment flavor
    and is contained in the specified model. If one of these conditions
    is not met, an exception is thrown.

    :param model_config: An MLflow Model object
    :param flavor: The deployment flavor to validate
    """
    if flavor not in SUPPORTED_DEPLOYMENT_FLAVORS:
        raise MlflowException(
            message=(
                "The specified flavor: `{flavor_name}` is not supported for deployment."
                " Please use one of the supported flavors: {supported_flavor_names}".format(
                    flavor_name=flavor,
                    supported_flavor_names=SUPPORTED_DEPLOYMENT_FLAVORS)),
            error_code=INVALID_PARAMETER_VALUE)
    elif flavor not in model_config.flavors:
        raise MlflowException(
            message=("The specified model does not contain the specified deployment flavor:"
                     " `{flavor_name}`. Please use one of the following deployment flavors"
                     " that the model contains: {model_flavors}".format(
                        flavor_name=flavor, model_flavors=model_config.flavors.keys())),
            error_code=RESOURCE_DOES_NOT_EXIST)


def deploy(model_key, model_uri, flavor=None, device='cpu', **kwargs):
    """
    Deploy an MLFlow model to RedisAI. User needs to pass the URL and credentials
    to connect to RedisAI server. Currently it accepts only TorchScript model, freezed
    Tensorflow model and SavedModel from tensorflow through MLFlow although RedisAI
    can takes Tensorflow lite model, ONNX model (any models like scikit-learn, spark
    which is converted to ONNX).

    Note: ml2rt is a package we have developed which can
        - do the conversion from different frameworks to ONNX
        - load SavedModel, freezed tensorflow, torchscript or ONNX models from disk
        - load script

    :param model_key: Redis Key on which we deploy the model
    :param model_uri: The location, in URI format, of the MLflow model to deploy to RedisAI.
                      For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param flavor: The name of the flavor of the model to use for deployment. Must be either
                   ``None`` or one of mlflow_redisai.pytorch.SUPPORTED_DEPLOYMENT_FLAVORS.
                   If ``None``, a flavor is automatically selected from the model's available
                   flavors. If the specified flavor is not present or not supported for deployment,
                   an exception will be thrown.
    :param device: GPU or CPU
    :param kwargs: Parameters for RedisAI connection

    """
    model_path = _download_artifact_from_uri(model_uri)
    # TODO: use os.path for python2.x compatiblity
    path = Path(model_path)
    model_config = path/'MLmodel'
    if not model_config.exists():
        raise MlflowException(
            message=(
                "Failed to find MLmodel configuration within the specified model's"
                " root directory."),
            error_code=INVALID_PARAMETER_VALUE)
    model_config = Model.load(model_config)

    if flavor is None:
        flavor = _get_preferred_deployment_flavor(model_config)
    else:
        _validate_deployment_flavor(model_config, flavor)
    _logger.info("Using the %s flavor for deployment!", flavor)

    con = redisai.Client(**kwargs)
    if flavor == mlflow.tensorflow.FLAVOR_NAME:
        tags = model_config.flavors[flavor]['meta_graph_tags']
        signaturedef = model_config.flavors[flavor]['signature_def_key']
        model_dir = path/model_config.flavors[flavor]['saved_model_dir']
        model, inputs, outputs = ml2rt.load_model(model_dir, tags, signaturedef)
    else:
        # TODO: this assumes the torchscript is saved using mlflow-redisai
        model_path = list(path.joinpath('data').iterdir())[0]
        if model_path.suffix != '.pt':
            raise RuntimeError("Model file does not have a valid suffix. Expected .pt")
        model = ml2rt.load_model(model_path)
        inputs = outputs = None
    try:
        device = redisai.Device.__members__[device]
    except KeyError:
        raise MlflowException(
            message="Invalid value for ``device``. It only accepts ``cpu`` or ``gpu``",
            error_code=INVALID_PARAMETER_VALUE)
    try:
        backend = _flavor2backend[flavor]
    except KeyError:
        raise MlflowException(
            message="Invalid value for ``backend``. It only accepts one of {}".format(
                _flavor2backend.keys()
            ),
            error_code=INVALID_PARAMETER_VALUE)
    backend = redisai.Backend.__members__[backend]
    con.modelset(model_key, backend, device, model, inputs=inputs, outputs=outputs)


def delete(model_key, **kwargs):
    """
    Delete a RedisAI model key and value.

    :param model_key: Redis Key on which we deploy the model
    """
    con = redisai.Client(**kwargs)
    con.modeldel(model_key)
    _logger.info("Deleted model with key: %s", model_key)

