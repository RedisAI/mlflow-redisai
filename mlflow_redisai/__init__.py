from pathlib import Path
import logging

import redisai
import ml2rt
from mlflow.deployments import BasePlugin
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
import mlflow.tensorflow

from .utils import get_preferred_deployment_flavor, SUPPORTED_DEPLOYMENT_FLAVORS, flavor2backend


logger = logging.getLogger(__name__)


class RedisAIPlugin(BasePlugin):
    def create(self, model_uri, flavor=None, **kwargs):
        model_path = _download_artifact_from_uri(model_uri)
        path = Path(model_path)
        model_config = path / 'MLmodel'
        if not model_config.exists():
            raise MlflowException(
                message=(
                    "Failed to find MLmodel configuration within the specified model's"
                    " root directory."),
                error_code=INVALID_PARAMETER_VALUE)
        model_config = Model.load(model_config)

        if flavor is None:
            flavor = get_preferred_deployment_flavor(model_config)
        else:
            self._validate_deployment_flavor(model_config, flavor, SUPPORTED_DEPLOYMENT_FLAVORS)
        logger.info("Using the {} flavor for deployment!".format(flavor))

        con = redisai.Client(**kwargs)
        if flavor == mlflow.tensorflow.FLAVOR_NAME:
            # TODO: test this for tf1.x and tf2.x
            tags = model_config.flavors[flavor]['meta_graph_tags']
            signaturedef = model_config.flavors[flavor]['signature_def_key']
            model_dir = path / model_config.flavors[flavor]['saved_model_dir']
            model, inputs, outputs = ml2rt.load_model(model_dir, tags, signaturedef)
        else:
            model_path = list(path.joinpath('data').iterdir())[0]
            # TODO: test this
            if model_path.suffix != '.pt':
                raise RuntimeError("Model file does not have a valid suffix. Expected ``.pt``")
            model = ml2rt.load_model(model_path)
            inputs = outputs = None
        device = kwargs.get('device', 'CPU')
        backend = flavor2backend[flavor]
        key = kwargs['modelkey']
        con.modelset(key, backend, device, model, inputs=inputs, outputs=outputs)
        return {'deployment_id': key, 'flavor': flavor}

    def delete(self, deployment_id, **kwargs):
        return None

    def update(self, deployment_id, model_uri=None, flavor=False, **kwargs):
        return {'flavor': flavor}

    def list(self, **kwargs):
        return ['f_deployment_id']

    def get(self, deployment_id, **kwargs):
        return {'key1': 'val1', 'key2': 'val2'}
