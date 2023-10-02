"""
Let's get the relationships yo
"""

import torch.nn.functional as F
import torch.nn.parallel
from models.kd.my_model_components import *
import math
from utils.logger import MeanLogger
from models.kd.feature_distill_zoo import *


@Model.register("ModelKDInfoNCE")
class ModelKDInfoNCE(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            backbone: AttentionQABackboneDetectorEncoder,
            attention_module_1: AttentionQAAttention,
            attention_module_2: AttentionQAAttention,
            reasoning_module_1: AttentionQAReasoning,
            reasoning_module_2: AttentionQAReasoning,
            temperature_kd: int = 1,
            temperature_infonce: int = 1,
            alpha: float = 0.5,
            initializer: InitializerApplicator = InitializerApplicator(),
    ):
        super().__init__(vocab)
        self.backbone = backbone
        self.attention_module_dict = torch.nn.ModuleDict({
            '1': attention_module_1,
            '2': attention_module_2,
        })
        self.reasoning_module_dict = torch.nn.ModuleDict({
            '1': reasoning_module_1,
            '2': reasoning_module_2,
        })
        self.infonce_comp = InfoNCEEmbLoss_2(temperature_infonce)
        # self.infonce_comp = InfoNCEEmbLoss_R(temperature_infonce)
        self.alpha = alpha  # kd loss的系数，相应ce loss的系数为 1 - alpha
        self.T = temperature_kd
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
            features_penult_qr2a: torch.Tensor = None,
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
        output_dict_attention = {}
        for task in ('1', '2'):
            output_dict_attention[task] = self.attention_module_dict[task](
                obj_reps=output_dict_backbone['obj_reps'],
                box_mask=output_dict_backbone['box_mask'],
                q_rep=output_dict_backbone[f'input_dict_{task}']['q_rep'],
                question_mask=output_dict_backbone[f'input_dict_{task}']['question_mask'],
                a_rep=output_dict_backbone[f'input_dict_{task}']['a_rep']
            )

        output_dict_reasoning = {}
        for task in ('1', '2'):
            output_dict_reasoning[task] = self.reasoning_module_dict[task](
                obj_reps=output_dict_backbone['obj_reps'],
                q_rep=output_dict_backbone[f'input_dict_{task}']['q_rep'],
                question_mask=output_dict_backbone[f'input_dict_{task}']['question_mask'],
                a_rep=output_dict_backbone[f'input_dict_{task}']['a_rep'],
                answer_mask=output_dict_backbone[f'input_dict_{task}']['answer_mask'],
                qa_attention_weights=output_dict_attention[task]['qa_attention_weights'],
                atoo_attention_weights=output_dict_attention[task]['atoo_attention_weights'],
                metadata=output_dict_backbone['metadata'],
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

        features_penult_qa2r = output_dict['features_penult_2']
        loss_infonce, infonce_info = self.infonce_comp(features_penult_qr2a, input_dict_1['label'], features_penult_qa2r, input_dict_2['label'])
        output_dict['loss_infonce'] = loss_infonce[None]
        output_dict['nce_t'] = infonce_info['nce_t'][None]
        output_dict['nce_s'] = infonce_info['nce_s'][None]
        output_dict['ncep_t'] = infonce_info['ncep_t'][None]
        output_dict['ncep_s'] = infonce_info['ncep_s'][None]
        output_dict['nceacc_t'] = infonce_info['nceacc_t'][None]
        output_dict['nceacc_s'] = infonce_info['nceacc_s'][None]

        output_dict['cnn_regularization_loss'] = output_dict_backbone['obj_reps']['cnn_regularization_loss']
        return output_dict

    def loss_kd(self, logits_t, logits_s):
        logp_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t = F.softmax(logits_t / self.T, dim=1)
        loss = F.kl_div(logp_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy_1': self.reasoning_module_dict['1'].get_metrics(reset)['accuracy'],
            'accuracy_2': self.reasoning_module_dict['2'].get_metrics(reset)['accuracy'],
        }
