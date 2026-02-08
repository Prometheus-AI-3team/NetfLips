from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart.model import BARTModel, bart_large_architecture
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch
from typing import Optional
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

@register_model("utut")
class UTUTModel(BARTModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        state = load_checkpoint_to_cpu("unit_mbart.pt")
        state_dict = state["model"]

        self_state_dict = self.state_dict()

        # state_dict["encoder.output_projection.weight"] = self_state_dict["encoder.output_projection.weight"]
        for w in [
            "encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
            "decoder.output_projection.weight",
        ]:
            vocab_size = state_dict[w].shape[0] - 4 - 2 - 1
            assert vocab_size == 1000
            state_dict[w] = torch.cat([state_dict[w][:4+vocab_size+2], self_state_dict[w][4+vocab_size+2:-1], state_dict[w][-1:]])

        self.load_state_dict(state_dict, strict=True)

@register_model_architecture("utut", "utut_large")
def mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)
