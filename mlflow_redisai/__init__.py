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
        key = kwargs.pop('modelkey')
        try:
            device = kwargs.pop('device')
        except KeyError:
            device = 'CPU'
        path = Path(_download_artifact_from_uri(model_uri))
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
            model_path = None
            for file in path.iterdir():
                if file.suffix == '.pt':
                    model_path = file
            if model_path is None:
                raise RuntimeError("Model file does not have a valid suffix. Expected ``.pt``")
            model = ml2rt.load_model(model_path)
            inputs = outputs = None
        backend = flavor2backend[flavor]
        con.modelset(key, backend, device, model, inputs=inputs, outputs=outputs)
        return {'deployment_id': key, 'flavor': flavor}

    def delete(self, deployment_id, **kwargs):
        """
        Delete a RedisAI model key and value.

       :param deployment_id: Redis Key on which we deploy the model
        """
        con = redisai.Client(**kwargs)
        con.modeldel(deployment_id)
        logger.info("Deleted model with key: {}".format(deployment_id))

    def update(self, deployment_id, model_uri=None, flavor=False, **kwargs):
        try:
            device = kwargs.pop('device')
        except KeyError:
            device = 'CPU'
        con = redisai.Client(**kwargs)
        try:
            con.modelget(deployment_id, meta_only=True)
        except Exception:  # TODO: check specificially for KeyError and raise MLFlowException with proper error code
            raise MlflowException("Model doesn't exist. If you trying to create new "
                                  "deployment, use ``create_deployment``")
        else:
            ret = self.create(model_uri, flavor, modelkey=deployment_id, device=device, **kwargs)
        return {'flavor': ret['flavor']}

    def list(self, **kwargs):
        # TODO: May be support RedisAI SCRIPT, eventually
        con = redisai.Client(**kwargs)
        return con.modelscan()

    def get(self, deployment_id, **kwargs):
        con = redisai.Client(**kwargs)
        return con.modelget(deployment_id, meta_only=True)
