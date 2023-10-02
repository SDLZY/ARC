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
from torch import nn

from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator

@Model.register("MultiHopAttentionQAReasoningBlock")
class AttentionQAReasoningBlock(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQAReasoningBlock, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.reasoning_encoder = TimeDistributed(reasoning_encoder)

        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        dim = sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        (span_encoder.get_output_dim(), self.pool_answer),
                                        (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            # torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        # self._accuracy = CategoricalAccuracy()
        # self._loss = torch.nn.CrossEntropyLoss()
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
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]

        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        # Now get the question representations
        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        ####################################
        # Perform Q by A attention
        # [batch_size, 4, question_length, answer_length]
        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))

        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1],
                                                            a_rep.shape[2], obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:,None,None])
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

        pooled_rep = replace_masked_values(things_to_pool,answer_mask[...,None], -1e7).max(2)[0]
        reasoning_inp = self.final_mlp(pooled_rep)
        output_dict = {
            'reasoning_inp': reasoning_inp,
            'cnn_regularization_loss': obj_reps['cnn_regularization_loss']
        }

        return output_dict


@Model.register("casual_r2c_gru_4cls")
class CasualR2CGRU4Cls(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 model_q2a: AttentionQAReasoningBlock,
                 model_qa2r: AttentionQAReasoningBlock,
                 qr2a: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super().__init__(vocab)
        self.model_q2a = model_q2a
        self.model_qa2r = model_qa2r
        self.qr2a = qr2a
        # TODO: 换激活函数
        self.reasoning_gru = nn.GRU(input_size=1024, hidden_size=512, batch_first=True, bidirectional=False)
        # self.reasoning_gru = nn.GRU(input_size=1024, hidden_size=512, batch_first=True, dropout=0.3, bidirectional=False)
        self.final_mlp_q2a = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
        )
        self.final_mlp_qa2r = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
        )
        self._accuracy_q2a = CategoricalAccuracy()
        self._accuracy_qa2r = CategoricalAccuracy()
        if qr2a:
            self._accuracy_qr2a = CategoricalAccuracy()
        # self._accuracy_q2ar = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self, batch_q2a, batch_qa2r, batch_qr2a=None):
        self.reasoning_gru.flatten_parameters()
        output_q2a = self.forward_q2a(**batch_q2a)
        batch_size = output_q2a['reasoning_inp'].shape[0]

        reasoning_inp_q2a = output_q2a['reasoning_inp'].view(batch_size*4, 1, -1)
        reasoning_hid_q2a, _ = self.reasoning_gru(reasoning_inp_q2a)
        reasoning_hid_q2a = reasoning_hid_q2a.squeeze(1)

        logits_q2a = self.final_mlp_q2a(reasoning_hid_q2a).view(batch_size, 4)
        class_probabilities_q2a = F.softmax(logits_q2a, dim=-1)
        output_dict_q2a = dict()
        output_dict_q2a['cnn_regularization_loss'] = output_q2a['cnn_regularization_loss']
        output_dict_q2a["label_logits"] = logits_q2a
        output_dict_q2a["label_probs"] = class_probabilities_q2a
        if 'label' in batch_q2a:
            loss_q2a = self._loss(logits_q2a, batch_q2a['label'].long().view(-1))
            self._accuracy_q2a(logits_q2a, batch_q2a['label'])
            output_dict_q2a['loss'] = loss_q2a[None]

        if 'label' in batch_q2a:
            label_q2a = batch_q2a['label']
        else:
            label_q2a = class_probabilities_q2a.argmax(-1)
        correct_reasoning_hid_q2a = reasoning_hid_q2a.view(batch_size, 4, -1)[torch.arange(batch_size), label_q2a]
        correct_reasoning_hid_q2a = correct_reasoning_hid_q2a.unsqueeze(1).repeat(1, 4, 1).view(1, batch_size*4, -1)

        output_qa2r = self.forward_qa2r(**batch_qa2r)
        reasoning_inp_qa2r = output_qa2r['reasoning_inp'].view(batch_size*4, 1, -1)
        reasoning_hid_qa2r, _ = self.reasoning_gru(reasoning_inp_qa2r, correct_reasoning_hid_q2a)
        reasoning_hid_qa2r = reasoning_hid_qa2r.squeeze(1)

        logits_qa2r = self.final_mlp_qa2r(reasoning_hid_qa2r).view(batch_size, 4)
        class_probabilities_qa2r = F.softmax(logits_qa2r, dim=-1)
        output_dict_qa2r = dict()
        output_dict_qa2r['cnn_regularization_loss'] = output_qa2r['cnn_regularization_loss']
        output_dict_qa2r["label_logits"] = logits_qa2r
        output_dict_qa2r["label_probs"] = class_probabilities_qa2r

        if 'label' in batch_qa2r:
            loss_qa2r = self._loss(logits_qa2r, batch_qa2r['label'].long().view(-1))
            self._accuracy_qa2r(logits_qa2r, batch_qa2r['label'])
            output_dict_qa2r['loss'] = loss_qa2r[None]

        if self.qr2a:
            if batch_qr2a is None:
                raise ValueError('Batch qr2a not defined !')
            output_qr2a = self.forward_qa2r(**batch_qr2a)
            reasoning_inp_qr2a = output_qr2a['reasoning_inp'].view(batch_size * 4, 1, -1)
            reasoning_hid_qr2a, _ = self.reasoning_gru(reasoning_inp_qr2a, reasoning_hid_q2a.unsqueeze(0))
            reasoning_hid_qr2a = reasoning_hid_qr2a.squeeze(1)

            logits_qr2a = self.final_mlp_qa2r(reasoning_hid_qr2a).view(batch_size, 4)
            class_probabilities_qr2a = F.softmax(logits_qr2a, dim=-1)
            output_dict_qr2a = dict()
            output_dict_qr2a['cnn_regularization_loss'] = output_qr2a['cnn_regularization_loss']
            output_dict_qr2a["label_logits"] = logits_qr2a
            output_dict_qr2a["label_probs"] = class_probabilities_qr2a

            if 'label' in batch_qr2a:
                loss_qr2a = self._loss(logits_qr2a, batch_qr2a['label'].long().view(-1))
                self._accuracy_qr2a(logits_qr2a, batch_qr2a['label'])
                output_dict_qr2a['loss'] = loss_qr2a[None]
        if self.qr2a:
            return output_dict_q2a, output_dict_qa2r, output_dict_qr2a
        else:
            return output_dict_q2a, output_dict_qa2r

    def forward_q2a(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        output_dict = self.model_q2a(*args, **kwargs)
        return output_dict

    def forward_qa2r(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        output_dict = self.model_qa2r(*args, **kwargs)
        return output_dict

    def get_metrics(self, reset=False):
        if self.qr2a:
            return {
                'accuracy_q2a': self._accuracy_q2a.get_metric(reset),
                'accuracy_qa2r': self._accuracy_qa2r.get_metric(reset),
                'accuracy_qr2a': self._accuracy_qr2a.get_metric(reset),
            }
        else:
            return {
                'accuracy_q2a': self._accuracy_q2a.get_metric(reset),
                'accuracy_qa2r': self._accuracy_qa2r.get_metric(reset),
                # 'accuracy': self._accuracy_q2ar.get_metric(reset),
            }
