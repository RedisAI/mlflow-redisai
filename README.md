# mlflow-redisai
[![Forum](https://img.shields.io/badge/Forum-RedisAI-blue)](https://forum.redislabs.com/c/modules/redisai)
[![Gitter](https://badges.gitter.im/RedisLabs/RedisAI.svg)](https://gitter.im/RedisLabs/RedisAI?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

A plugin that integrates RedisAI with MLflow pipeline. ``mlflow_redisai`` enables you to
use mlflow to deploy the models built and trained in mlflow pipeline into RedisAI without any
extra effort from the user. This plugin provides few command line APIs, which is also accessible
through mlflow's python package, to make the deployment process seamless.

## Installation
For installing and activating the plugin, you only need to install this package which is available
in pypi and can be installed with

```bash
pip install mlflow_redisai
```

## What does it do
Installing this package uses python's amazing entrypoint mechanism to register the plugin into MLflow's
plugin registry. This registry will be invoked each time you launch MLflow script or command line
argument.

## Options
This plugin allows you to interact with RedisAI deployment through MLflow using below given options.
All of these options are accessible through command line and python API, although the predict command
line option is not reliable in the current release. if you are connecting to a non-local RedisAI instance
or if your RedisAI instance needs non-default connection parameters such as username or password, take a
look at the [connection parameters section](#connection-parameters)

### Create deployment
Deploy the model to RedisAI. The `create` command line argument and ``create_deployment`` python
APIs does the deployment of a model built with MLflow to RedisAI. It fetches the information, such as
which framework the model is built on, from the model configuration file implicitly.

##### CLI
```shell script
mlflow deployments create -t redisai --name <rediskey> -m <model uri> -C <config option>
```

##### Python API
```python
from mlflow.deployments import get_deploy_client
target_uri = 'redisai'  # host = localhost, port = 6379
redisai = get_deploy_client(target_uri)
redisai.create_deployment(rediskey, model_uri, config={'device': 'GPU'})
```

### Update deployment
Update deployment API has a very similar signature to the create API. In face, update deployment
does exactly the same operation as create API except that the model should already be deployed.
If the model is not present already, it raises an exception. Update API can be used to update
a new model after retraining. This type of setup is useful if you want to change the device on which
the inference is running or if you want to change the autobatching size, or even if you are doing live 
training and updating the model on the fly . RedisAI will make sure the user experience is seamless
while changing the model in a live environment.

##### CLI
```shell script
mlflow deployments update -t redisai --name <rediskey> -m <model uri> -C <config option>
```

##### Python API
```python
redisai.update_deployment(rediskey, model_uri, config={'device': 'GPU'})
```

### Delete deployment
Delete an existing deployment. Error will be thrown if the model is not already deployed

##### CLI
```shell script
mlflow deployments delete -t redisai --name <rediskey>
```

##### Python API
```python
redisai.delete_deployment(rediskey)
```

### List all deployments
List the names of all the deployments. This name can then be used in other APIs or can be
used in the get deployment API to get more details about a particular deployment. Currently,
it displays every deployment, not just the deployment made through this plugin

##### CLI
```shell script
mlflow deployments list -t redisai
```

##### Python API
```python
redisai.list_deployments()
```

### Get deployment details
Get API fetches the meta data about a deployment from RedisAI. This metadata includes

- BACKEND : the backend used by the model as a String
- DEVICE : the device used to execute the model as a String
- TAG : the model's tag as a String
- BATCHSIZE : The maximum size of any batch of incoming requests. If BATCHSIZE is equal to 0 each incoming request is served immediately. When BATCHSIZE is greater than 0, the engine will batch incoming requests from multiple clients that use the model with input tensors of the same shape.
- MINBATCHSIZE : The minimum size of any batch of incoming requests.
- INPUTS : array reply with one or more names of the model's input nodes (applicable only for TensorFlow models)
- OUTPUTS : array reply with one or more names of the model's output nodes (applicable only for TensorFlow models)

##### CLI
```shell script
mlflow deployments get -t redisai --name <rediskey>
```

##### Python API
```python
redisai.get_deployment(rediskey)
```

### Plugin help
MLflow integration also made a handy help API which is specific to any plugins you install rather
than the help string of the mlflow command itself. For quickly checking something out, it'd be
easy to use the help API rather than looking out for this document.

PS: Since help is specific to the plugin and not really an attribute of the client object itself,
it's not available under client object (`redisai` variable in the above examples)

##### CLI
```shell script
mlflow deployments help -t redisai
``` 

### Local run
If you are new to RedisAI and trying it out for the first time, you might not know the setup already
(although it's quite easy to setup a local RedisAI instance). The `local-run` API is there to help
you, if that's the case. It pulls the latest docker image of RedisAI (yes, you need docker installed in your
machine for this to work), run it on the default port, deploy the model you specified.

PS: It's IMPORTANT to note that this API leaves the docker container running. You would need to manually
stop the container once you are done with experimentation. Also, remember that if you trying to run
this API twice without killing the first container, it throws an error saying the port is already
in use.

##### CLI
```shell script
mlflow deployments run-local -t redisai --name <rediskey> -m <model uri> -C <config option>
```


## Connection Parameters
For connecting to a RedisAI instance with non-default connection parameters, you'd either need to
supply the necessary (and allowed) parameters through environmental variables or modify the target
URI that you'd pass through the command line using `-t` or `-target` option.

##### Through environmental variables

Currently this plugin allows five options through environmental variables which are given below
(without description because they are self explanatory)

* REDIS_HOST
* REDIS_PORT
* REDIS_DB
* REDIS_USERNAME
* REDIS_PASSWORD

##### Modifying URI
A template for quick verification is given below
```
redisai:/[[username]:[password]]@[host]:[port]/[db]
```

where the default value each would be
- username = None
- password = None
- host = localhost
- port = 6379
- db = 0



