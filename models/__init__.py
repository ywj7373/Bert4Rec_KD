from .bert import BERTModel, SmallBERTModel

MODELS = {
    BERTModel.code(): BERTModel,
    SmallBERTModel.code(): SmallBERTModel
}


def model_factory(model_name, args):
    model = MODELS[model_name]
    return model(args)
