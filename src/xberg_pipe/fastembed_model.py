"""Ruri ONNXモデルのFastEmbed設定。"""

from xberg_pipe.keywording import DEFAULT_MODEL


def create_ruri_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """RuriのONNXモデルを登録して読み込む。"""

    if model_name != DEFAULT_MODEL:
        raise ValueError(f"未対応の埋め込みモデルです: {model_name}")

    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    if model_name not in {
        model["model"] for model in TextEmbedding.list_supported_models()
    }:
        TextEmbedding.add_custom_model(
            model=DEFAULT_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=DEFAULT_MODEL),
            dim=256,
            model_file="onnx/model_int8.onnx",
        )

    if device is None or device == "cpu":
        return TextEmbedding(model_name=model_name)
    if device == "cuda":
        return TextEmbedding(model_name=model_name, providers=["CUDAExecutionProvider"])
    if device.startswith("cuda:") and device[5:].isdigit():
        return TextEmbedding(
            model_name=model_name,
            providers=[("CUDAExecutionProvider", {"device_id": int(device[5:])})],
        )
    raise ValueError(f"未対応のデバイスです: {device}")
