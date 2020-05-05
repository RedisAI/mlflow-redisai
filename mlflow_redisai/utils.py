import logging
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
import mlflow.torchscript
import mlflow.tensorflow


# TODO: lazy loader

logger = logging.getLogger(__name__)
SUPPORTED_DEPLOYMENT_FLAVORS = [mlflow.torchscript.FLAVOR_NAME, mlflow.tensorflow.FLAVOR_NAME]
flavor2backend = {
    mlflow.torchscript.FLAVOR_NAME: 'torch',
    mlflow.tensorflow.FLAVOR_NAME: 'tf'}


def get_preferred_deployment_flavor(model_config):
    """
    Obtains the flavor that MLflow would prefer to use when deploying the model on RedisAI.
    If the model does not contain any supported flavors for deployment, an exception
    will be thrown.

    :param model_config: An MLflow model object
    :return: The name of the preferred deployment flavor for the specified model
    """
    # TODO: add onnx & TFlite
    possible_flavors = set(SUPPORTED_DEPLOYMENT_FLAVORS).intersection(model_config.flavors)
    if len(possible_flavors) == 1:
        return possible_flavors.pop()
    elif len(possible_flavors) > 1:
        flavor = possible_flavors.pop()
        logger.info("Found more than one possible flavors, using "
                    "the first: {}".format(flavor))
        return flavor
    else:
        raise MlflowException(
            message=(
                "The specified model does not contain any of the supported flavors for"
                " deployment. The model contains the following flavors: {model_flavors}."
                " Supported flavors: {supported_flavors}".format(
                    model_flavors=model_config.flavors.keys(),
                    supported_flavors=SUPPORTED_DEPLOYMENT_FLAVORS)),
            error_code=RESOURCE_DOES_NOT_EXIST)
