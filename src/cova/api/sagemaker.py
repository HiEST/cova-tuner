"""This module implements functions related to the usage of AWS Sagemaker"""

import json
import logging
import time

import sagemaker
from sagemaker.session import Session

# from sagemaker import ModelPackage

logger = logging.getLogger(__name__)


class ModelPackageArnProvider:
    """This class provides ARNs to SSD and YOLOv3 models for different regions of AWS Sagemaker."""

    @staticmethod
    def get_yolov3_model_package_arn(current_region: str) -> str:
        """Returns ARN for YOLOv3 model in the specified region.

        Args:
            current_region (str): AWS region

        Returns:
            str: ARN for YOLOv3 in the specified region.
        """
        mapping = {
            "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
        }
        return mapping[current_region]

    @staticmethod
    def get_ssd_model_package_arn(current_region: str) -> str:
        """Returns ARN for SSD-Resnet50 model in the specified region.

        Args:
            current_region (str): AWS region

        Returns:
            str: ARN for SSD-Resnet50 in the specified region.
        """
        mapping = {
            "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
        }
        return mapping[current_region]


def deploy_model(
    role,
    num_instances,
    model_arn,
    instance_type,
    model_name,
    output_path,
    max_concurrent_transforms=2,
):
    model = sagemaker.ModelPackage(
        role=role, model_package_arn=model_arn, sagemaker_session=sagemaker.Session()
    )
    # model.deploy(num_instances, instance_type, endpoint_name=model_name)
    transformer = model.transformer(
        instance_count=num_instances,
        instance_type=instance_type,
        output_path=output_path,
        max_concurrent_transforms=max_concurrent_transforms,
    )

    return model, transformer


def batch_transform(data, transformer, batch_output, content_type):
    ts0 = time.time()
    transformer.transform(
        data=data,
        data_type="S3Prefix",
        content_type=content_type,
        input_filter="$",
        join_source="None",
        output_filter="$",
    )
    ts_create = time.time() - ts0

    ts0 = time.time()
    transformer.wait()
    ts_exec = time.time() - ts0
    logger.info(
        f"Batch Transform job created in {ts_create:.2f} seconds and executed in {ts_exec:.2f} seconds."
    )

    assert batch_output == transformer.output_path
    output = transformer.output_path

    return output


def invoke_DL_endpoint(
    image_path, runtime, endpoint_name, content_type="image/png", bounding_box="no"
):
    img = open(image_path, "rb").read()

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=bytearray(img),
        ContentType=content_type,
        CustomAttributes='{"threshold": 0.2}',
        Accept="json",
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    return result


def get_default_bucket() -> str:
    """Returns default bucket of the Sagemaker session.

    Returns:
        str: default bucket in s3 of the Sagemaker session.
    """
    return Session().default_bucket()
