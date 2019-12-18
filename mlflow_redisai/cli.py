import click
import mlflow_redisai
from mlflow.utils import cli_args


@click.group("redisai")
def main():
    """
    Serve models on RedisAI.
    """
    pass


@main.command("deploy")
@click.option("--model-key", "-a", help="Model key in RedisAI", required=True)
@cli_args.MODEL_URI
@click.option("--host", default='localhost', help="RedisAI host address")
@click.option("--port", default="6379",
              help="RedisAI host port")
@click.option("--device", "-d", default='cpu', help="GPU or CPU")
@click.option("--flavor", "-f", default=None,
              help=("The name of the flavor to use for deployment. Must be one of the following:"
                    " {supported_flavors}. If unspecified, a flavor will be automatically selected"
                    " from the model's available flavors.".format(
                      supported_flavors=mlflow_redisai.SUPPORTED_DEPLOYMENT_FLAVORS)))
def deploy(model_key, model_uri, host, port, device, flavor):
    # TODO: add note about how to save the model because it doesn't accept
    # all MLFlow models
    """
    Deploy MLFlow models on RedisAI
    """
    mlflow_redisai.deploy(
        model_key=model_key,
        model_uri=model_uri,
        host=host,
        port=port,
        device=device,
        flavor=flavor)