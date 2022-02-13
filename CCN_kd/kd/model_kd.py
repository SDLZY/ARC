"""
Let's get the relationships yo
"""

import torch.nn.functional as F
import torch.nn.parallel
from kd.my_model_components import *
import math


@Model.register("ModelKD")
class ModelKD(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            backbone: AttentionQABackboneDetectorEncoder,
            reasoning_module_1: AttentionQAReasoning,
            reasoning_module_2: AttentionQAReasoning,
            temperature: int = 1,
            alpha: float = 0.5,
            initializer: InitializerApplicator = InitializerApplicator(),
    ):
        super().__init__(vocab)
        self.backbone = backbone
        self.reasoning_module_dict = torch.nn.ModuleDict({
            '1': reasoning_module_1,
            '2': reasoning_module_2,
        })
        self.T = temperature
        self.alpha = alpha
        initializer(self)

    def forward(
            self,
            images: torch.Tensor,
            objects: torch.LongTensor,
            segms: torch.Tensor,
            boxes: torch.Tensor,
            box_mask: torch.LongTensor,
            input_dict_1: Dict,
            input_dict_2: Dict,
            input_dict_4: Dict=None,
            logits_qr2a: torch.Tensor = None,
            metadata: List[Dict[str, Any]] = None):
        output_dict_backbone = self.backbone(
            images=images,
            objects=objects,
            segms=segms,
            boxes=boxes,
            box_mask=box_mask,
            input_dict_1=input_dict_1,
            input_dict_2=input_dict_2,
            input_dict_4=input_dict_4,
            metadata=metadata,
        )

        output_dict_reasoning = {}
        for task in ('1', '2'):
            output_dict_reasoning[task] = self.reasoning_module_dict[task](
                obj_reps=output_dict_backbone['obj_reps'],
                obj_feas=output_dict_backbone['obj_feas'],
                imggraph=output_dict_backbone['imggraph'],
                q_rep=output_dict_backbone[f'input_dict_{task}']['q_rep'],
                a_rep=output_dict_backbone[f'input_dict_{task}']['a_rep'],
                question_mask=output_dict_backbone[f'input_dict_{task}']['question_mask'],
                answer_mask=output_dict_backbone[f'input_dict_{task}']['answer_mask'],
                metadata=metadata,
                label=output_dict_backbone[f'input_dict_{task}'].get('label', None),
            )
        output_dict = {}
        for task in ('1', '2'):
            for key, value in output_dict_reasoning[task].items():
                output_dict[f'{key}_{task}'] = value
        logits_q2a = output_dict["label_logits_1"]
        loss_kd = self.loss_kd(logits_qr2a, logits_q2a)
        output_dict['loss_kd'] = loss_kd[None]
        # loss_kd = self.loss_kd(logits_qr2a, logits_q2a) * self.alpha
        # output_dict['loss_kd'] = loss_kd[None]
        # output_dict['loss_1'] *= (1 - self.alpha)
        output_dict['cnn_regularization_loss'] = output_dict_backbone['obj_reps']['cnn_regularization_loss']
        return output_dict

    def loss_kd(self, logits_t, logits_s):
        logp_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t = F.softmax(logits_t / self.T, dim=1)
        loss = F.kl_div(logp_s, p_t, reduction='mean') * (self.T ** 2)
        return loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy_1': self.reasoning_module_dict['1'].get_metrics(reset)['accuracy'],
            'accuracy_2': self.reasoning_module_dict['2'].get_metrics(reset)['accuracy'],
        }
