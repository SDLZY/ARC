"""
Let's get the relationships yo
"""

from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator


def cosine_sim(x, w, eps=1e-8):
    ip = torch.matmul(x, w).squeeze(-1)
    nx = torch.norm(x, 2, 2)
    nw = torch.norm(w, 2, 0)
    return ip / (nx * nw).clamp(min=eps)

@Model.register("MultiHopAttentionQABackboneDetectorEncoder")
class AttentionQABackboneDetectorEncoder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 class_embs: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQABackboneDetectorEncoder, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        initializer(self)

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                input_dict_0: Dict = None,
                input_dict_1: Dict = None,
                input_dict_2: Dict = None,
                input_dict_4: Dict = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        tasks = []
        if input_dict_0 is not None:
            tasks.append('0')
        if input_dict_1 is not None:
            tasks.append('1')
        if input_dict_2 is not None:
            tasks.append('2')
        if input_dict_4 is not None:
            tasks.append('4')

        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]
        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        # Now get the question representations
        output_dict = {
            "obj_reps": {
                'obj_reps': obj_reps['obj_reps'],
                'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
            },
            "images": images,
            "objects": objects,
            "segms": segms,
            "boxes": boxes,
            "box_mask": box_mask,
            "input_dict_0": input_dict_0,
            "input_dict_1": input_dict_1,
            "input_dict_2": input_dict_2,
            "input_dict_4": input_dict_4,
            "metadata": metadata,
        }
        for task in tasks:
            question = output_dict[f'input_dict_{task}']['question']
            question_tags = output_dict[f'input_dict_{task}']['question_tags']
            question_mask = output_dict[f'input_dict_{task}']['question_mask']
            q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])

            answers = output_dict[f'input_dict_{task}']['answers']
            answer_tags = output_dict[f'input_dict_{task}']['answer_tags']
            answer_mask = output_dict[f'input_dict_{task}']['answer_mask']
            a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

            output_dict[f'input_dict_{task}']['q_rep'] = q_rep
            output_dict[f'input_dict_{task}']['q_obj_reps'] = q_obj_reps
            output_dict[f'input_dict_{task}']['a_rep'] = a_rep
            output_dict[f'input_dict_{task}'][f'a_obj_reps'] = a_obj_reps

        return output_dict


@Model.register("MultiHopAttentionQAAttention")
class AttentionQAAttention(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQAAttention, self).__init__(vocab)
        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        initializer(self)

    def forward(self, obj_reps, box_mask, q_rep, question_mask, a_rep):
        output_dict = {}
        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        output_dict['qa_similarity'] = qa_similarity
        output_dict['qa_attention_weights'] = qa_attention_weights

        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1],
                                                                        a_rep.shape[2], obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:, None, None])
        output_dict['atoo_similarity'] = atoo_similarity
        output_dict['atoo_attention_weights'] = atoo_attention_weights
        return output_dict


@Model.register("MultiHopAttentionQAReasoning")
class AttentionQAReasoning(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 reasoning_use_obj: bool = True,
                 reasoning_use_answer: bool = True,
                 reasoning_use_question: bool = True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 reduction: str = 'mean',
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQAReasoning, self).__init__(vocab)
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.reasoning_encoder = TimeDistributed(reasoning_encoder)

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        dim = sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        (512, self.pool_answer),
                                        (512, self.pool_question)] if to_pool])

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self._accuracy = CategoricalAccuracy()
        self.reduction = reduction
        self._loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        initializer(self)

    def forward(self, obj_reps, q_rep, question_mask, a_rep, answer_mask, qa_attention_weights, atoo_attention_weights,
                metadata: List[Dict[str, Any]] = None, label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        output_dict = {}
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))

        reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
                                                        (attended_o, self.reasoning_use_obj),
                                                        (attended_q, self.reasoning_use_question)]
                                   if to_pool], -1)

        if self.rnn_input_dropout is not None:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)
        reasoning_output = self.reasoning_encoder(reasoning_inp, answer_mask)

        ###########################################
        things_to_pool = torch.cat([x for x, to_pool in [(reasoning_output, self.pool_reasoning),
                                                         (a_rep, self.pool_answer),
                                                         (attended_q, self.pool_question)] if to_pool], -1)

        pooled_rep = replace_masked_values(things_to_pool, answer_mask[..., None], -1e7).max(2)[0]
        # logits = self.final_mlp(pooled_rep).squeeze(2)
        # final_rep = self.final_mlp[:-1](pooled_rep)
        # cosine = cosine_sim(final_rep, self.final_mlp[-1].weight.t())
        # logits = self.final_mlp[-1](final_rep).squeeze(2)

        output_dict['features_penult'] = pooled_rep
        # output_dict['features_last'] = self.final_mlp[:2](pooled_rep)
        # logits = self.final_mlp[2:](output_dict['features_last']).squeeze(2)
        logits = self.final_mlp(pooled_rep).squeeze(2)
        ###########################################

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict.update({"label_logits": logits, "label_probs": class_probabilities,
                            'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                            # 'label_cosine': cosine
                            # Uncomment to visualize attention, if you want
                            # 'qa_attention_weights': qa_attention_weights,
                            # 'atoo_attention_weights': atoo_attention_weights,
                            })
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            if self.reduction == 'mean':
                output_dict["loss"] = loss[None]
            else:
                output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}


