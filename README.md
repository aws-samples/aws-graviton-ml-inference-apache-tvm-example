# Run ML inference on EC2 (Graviton) using Apache TVM

This sample provides steps to deploy Apache TVM (TVMC Python) on a Graviton (arm64)
 EC2 instance to do ML inference using a ResNet50 model (ONNX).

## Overview

[Apache TVM](https://github.com/apache/tvm) is an open-source ML compiler framework
 for optimizing and running computations efficiently on various hardware backends.

It is one of the options available to perform CPU-based ML inference
 on [Graviton](https://aws.amazon.com/ec2/graviton/) instances.

## Architecture

One EC2 instance (AWS Graviton) with Linux, running a Docker container with TVMC.

There are 2 Python scripts to run inside the container in order:

1. **Tune the ML model** on the specific hardware
1. **Run ML inference** using the tuning records

## Instructions

### Step 1. Create EC2 instance

Go to service Amazon EC2.
* [Create an EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)
 of type **c7g.large** (or larger for faster build and tuning times) with OS **Ubuntu 20.04** built
 for the **arm64** architecture.
* The EBS volume should have a minimum of **20GiB**.

### Step 2. Connect to EC2 instance

Connect to the EC2 instance using your preferred method from these
 [instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html).
* Install Docker using these [instructions](https://docs.docker.com/engine/install/ubuntu/)
 and ensure you can use Docker as a non-root user
* Ensure current directory (`pwd`) is */home/ubuntu*. Otherwise, run `cd /home/ubuntu`
* Run `git clone https://github.com/aws-samples/aws-graviton-ml-inference-apache-tvm-example` to
 have all sample files available locally on the EC2 instance. Or use an alternative method to
 copy the files to the EC2 instance (e.g. `scp`)
* Run `cd aws-graviton-ml-inference-apache-tvm-example`

### Step 3. Build Docker container image

* Run `docker build src -t tvmc`

### Step 4. Tune ML model

* Run `docker run -v ~/aws-graviton-ml-inference-apache-tvm-example:/sample -it tvmc /usr/bin/python3 /sample/src/tune.py`

Refer to the tuning section in the
 [TVMC tutorial](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html#automatically-tuning-the-resnet-model).

### Step 5. Run ML Inference

* Run `docker run -v ~/aws-graviton-ml-inference-apache-tvm-example:/sample -it tvmc /usr/bin/python3 /sample/src/app.py`

Confirm the result matches this output:

```
[
    {'class': 'n02123159 tiger cat', 'prob': 0.58619...},
    {'class': 'n02123045 tabby, tabby cat', 'prob': 0.30448...},
    {'class': 'n02124075 Egyptian cat', 'prob': 0.08718...},
    {'class': 'n02129604 tiger, Panthera tigris', 'prob': 0.00323...},
    {'class': 'n02128385 leopard, Panthera pardus', 'prob': 0.00179...}
]
```

## References

* Apache TVM: https://github.com/apache/tvm
* TVMC driver: https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html
* TVM auto-scheduler: https://tvm.apache.org/2021/03/03/intro-auto-scheduler
* ONNX model: https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-7.onnx
* ONNX model labels: https://s3.amazonaws.com/onnx-model-zoo/synset.txt

## Contributing

See [CONTRIBUTING](./CONTRIBUTING.md) for more information.

## License

This sample code is made available under a MIT-0 license. See the [LICENSE](./LICENSE) file.
