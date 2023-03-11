"""
The components of CCN
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
from utils.newdetector import SimpleDetector
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
                 input_dropout: float = 0.5,
                 class_embs: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQABackboneDetectorEncoder, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)

        self.scene_reps = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, 1),
            torch.nn.MaxPool2d(2, stride=2)
        )

        self.img_graph0 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Sigmoid(),
        )
        self.img_graph1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Tanh(),
        )

        self.obj_graph0 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Sigmoid(),
        )
        self.obj_graph1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Tanh(),
        )

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

    def img_graph(self, img_feats):
        N, C, H, W = img_feats.shape[0:]
        x = img_feats.view(N, C, -1).permute(0, 2, 1)
        num = x.shape[1]
        diag = torch.ones(num)
        diag = torch.diag(diag)
        scene_graph = torch.matmul(x, x.permute(0, 2, 1))

        ones = torch.ones(num, num)
        diag_zero = ones - diag
        scene_graph = masked_softmax(scene_graph, diag_zero[None, ...].cuda(), dim=1) + diag[None, ...].cuda()

        scene_value = torch.matmul(scene_graph, x)[:, :, None]
        scene_conv1 = self.img_graph0(scene_value.permute(0, 3, 1, 2))
        scene_conv2 = self.img_graph1(scene_value.permute(0, 3, 1, 2))
        scene_conv = scene_conv1 * scene_conv2
        scene_c = scene_conv.squeeze(-1)
        scene = scene_c.view(N, scene_c.shape[1], H, W)
        return scene

    def obj_graph(self, obj_reps):
        B, N, C = obj_reps.shape[0:]
        x = obj_reps
        diag = torch.ones(N)
        diag = torch.diag(diag)
        object_graph = torch.matmul(x, x.permute(0, 2, 1))

        ones = torch.ones(N, N)
        diag_zero = ones - diag
        object_graph = masked_softmax(object_graph, diag_zero[None, ...].cuda(), dim=1) + diag[None, ...].cuda()

        obj_value = torch.matmul(object_graph, x)[:, :, None]
        obj_conv1 = self.obj_graph0(obj_value.permute(0, 3, 1, 2))
        obj_conv2 = self.obj_graph1(obj_value.permute(0, 3, 1, 2))
        obj_conv = obj_conv1 * obj_conv2
        obj_c = obj_conv.squeeze(-1).permute(0, 2, 1)
        return obj_c

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
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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

        obj_reps, img_feats = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        obj_feas = self.obj_graph(obj_reps['obj_reps'])

        scene = self.scene_reps(img_feats)
        imggraph = self.img_graph(scene)

        output_dict = {
            "obj_reps": {
                'obj_reps': obj_reps['obj_reps'],
                'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
            },
            "img_feats": img_feats,
            "obj_feas": obj_feas,
            "scene": scene,
            "imggraph": imggraph,
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
            q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_feas)

            answers = output_dict[f'input_dict_{task}']['answers']
            answer_tags = output_dict[f'input_dict_{task}']['answer_tags']
            answer_mask = output_dict[f'input_dict_{task}']['answer_mask']
            a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_feas)

            output_dict[f'input_dict_{task}']['q_rep'] = q_rep
            output_dict[f'input_dict_{task}']['q_obj_reps'] = q_obj_reps
            output_dict[f'input_dict_{task}']['a_rep'] = a_rep
            output_dict[f'input_dict_{task}'][f'a_obj_reps'] = a_obj_reps

        return output_dict


@Model.register("MultiHopAttentionQAReasoning")
class AttentionQAReasoning(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.5,
                 hidden_dim_maxpool: int = 512,
                 num_cluster: int = 32,
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

        self.rnn_input_dropout = TimeDistributed(
            InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.reasoning_encoder = TimeDistributed(reasoning_encoder)

        self.qnode_graph11 = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 3, 512, 1),
            torch.nn.Sigmoid(),
        )
        self.qnode_graph12 = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 3, 512, 1),
            torch.nn.Tanh(),
        )

        self.anode_graph11 = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 3, 512, 1),
            torch.nn.Sigmoid(),
        )
        self.anode_graph12 = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 3, 512, 1),
            torch.nn.Tanh(),
        )

        self.netvlad = torch.nn.Sequential(
            torch.nn.Conv2d(512, 32, 1, bias=True)
        )

        self.vladchannel = torch.nn.Sequential(
            torch.nn.Linear(1024, 512)
        )

        self.scene_graph0 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Sigmoid(),
        )
        self.scene_graph1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
            torch.nn.Tanh(),
        )

        self.dropout = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
        )

        self.centroids = torch.nn.Parameter(torch.rand(num_cluster, 512))
        self.num_cluster = num_cluster

        self.gama = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, 1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(512, 512, 1),
        )

        self.fusion_conv = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
        )

        self.question_conv = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
        )

        self.answer_conv = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 1),
        )

        self.reason_conv = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 2, 512, 1),
        )
        self.reasoning1 = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 2, 512, 1),
            torch.nn.Sigmoid(),
        )
        self.reasoning2 = torch.nn.Sequential(
            torch.nn.Conv2d(512 * 2, 512, 1),
            torch.nn.Tanh(),
        )

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(1024 + 512, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(1024, 1),
        )
        self._accuracy = CategoricalAccuracy()
        self.reduction = reduction
        self._loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)
        initializer(self)

    def vlad(self, scene, q_final):
        x = scene
        N, C, W, H = x.shape[0:]

        q_rep = q_final[:, :, None, None].repeat(1, 1, W, H)
        s_q = torch.cat([scene, q_rep], 1)
        gama = self.gama(s_q)
        gama = gama.permute(0, 2, 3, 1).reshape(N, W * H, C).permute(0, 2, 1)

        x = F.normalize(x, p=2, dim=1)
        soft_assign = self.netvlad(x)

        soft_assign = F.softmax(soft_assign, dim=1)
        soft_assign = soft_assign.view(soft_assign.shape[0], soft_assign.shape[1], -1)

        x_flatten = x.view(N, C, -1)

        x1 = x_flatten.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)
        x2 = self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        gama_new = gama.expand(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3)
        x2 = gama_new * x2

        residual = x1 - x2
        residual = residual * soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        newvlad = torch.max(x2, 3, keepdim=False)[0]

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        vlad = vlad.view(vlad.shape[0], self.num_cluster, C)
        vlad = torch.cat((vlad, newvlad), -1)
        vlad = self.vladchannel(vlad)
        return vlad

    def reason_answer(self, reps):
        src = reps.view(reps.shape[0] * reps.shape[1], reps.shape[2], reps.shape[3])
        x1 = self.reason_conv(reps.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        B, N, L, C = x1.shape[0:]
        x = x1.reshape(B * N, L, C)
        diag = torch.ones(L)
        diag = torch.diag(diag)
        graph = torch.matmul(x, x.permute(0, 2, 1))
        direct = torch.sign(graph)
        length = torch.abs(graph)

        ones = torch.ones(L, L)
        diag_zero = ones - diag
        length = masked_softmax(length, diag_zero[None, ...].cuda(), dim=1)

        direct_graph = direct * length + diag[None, ...].cuda()

        direct_value = torch.matmul(direct_graph, src)[:, :, None]
        direct_conv1 = self.reasoning1(direct_value.permute(0, 3, 1, 2))
        direct_conv2 = self.reasoning2(direct_value.permute(0, 3, 1, 2))

        direct_conv = direct_conv1 * direct_conv2

        result = direct_conv.squeeze(-1).permute(0, 2, 1)
        result = result.view(B, N, direct_conv.shape[2], direct_conv.shape[1])
        return result

    def forward(self,
                obj_reps, obj_feas, imggraph,
                q_rep, a_rep, question_mask, answer_mask,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        MASK = torch.sum(question_mask, 2, keepdim=True) - 1
        onehot = torch.cuda.LongTensor(MASK.size(0), q_rep.size(2), MASK.size(1)).zero_()
        target = onehot.scatter_(1, MASK.long().permute(0, 2, 1), 1).permute(0, 2, 1).float()
        q_final = torch.sum(q_rep * target[..., None], 2, keepdim=False)
        q_final = torch.mean(q_final, 1, keepdim=False)  ## 32 * 512

        scene = self.vlad(imggraph, q_final)

        diag = torch.ones(self.num_cluster)
        diag = torch.diag(diag)
        scene_graph = torch.matmul(scene, scene.permute(0, 2, 1))

        ones = torch.ones(self.num_cluster, self.num_cluster)
        diag_zero = ones - diag
        scene_graph = masked_softmax(scene_graph, diag_zero[None, ...].cuda(), dim=1) + diag[None, ...].cuda()

        scene_value = torch.matmul(scene_graph, scene)[:, :, None]
        scene_conv1 = self.scene_graph0(scene_value.permute(0, 3, 1, 2))
        scene_conv2 = self.scene_graph1(scene_value.permute(0, 3, 1, 2))
        scene_conv = scene_conv1 * scene_conv2
        scene_c = scene_conv.squeeze(-1)
        scene_c = self.dropout(scene_c)
        scene = scene_c

        ####################################
        # Perform scene attention by question
        q_rep1 = self.question_conv(q_rep.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        scene_attend = torch.matmul(q_rep1, scene_c[:, None])
        scene_attention_weights = masked_softmax(scene_attend, question_mask[..., None])
        scene_qo = torch.einsum('bnao,bod->bnad', (scene_attention_weights, scene.permute(0, 2, 1)))

        obj_attend = torch.matmul(q_rep, obj_feas.permute(0, 2, 1)[:, None])
        obj_attention_weights = masked_softmax(obj_attend, question_mask[..., None])
        obj_qo = torch.einsum('bnao,bod->bnad', (obj_attention_weights, obj_feas))

        question_node = torch.cat([scene_qo, obj_qo, q_rep], -1)

        # question_node = self.dropout(question_node)

        ## question first layer
        question_diag = torch.diag_embed(
            question_mask.view(question_mask.shape[0] * question_mask.shape[1], question_mask.shape[2]))
        ques_diag = question_diag.view(question_mask.shape[0], question_mask.shape[1], question_mask.shape[2],
                                       question_mask.shape[2]).float()
        first_graph = torch.matmul(question_node, question_node.permute(0, 1, 3, 2))
        first_mask = torch.matmul(question_mask[..., None].float(), question_mask[:, :, None, :].float())
        first_graph = masked_softmax(first_graph, first_mask, dim=2) + ques_diag

        first_conv = torch.matmul(first_graph, question_node)
        first_conv1 = self.qnode_graph11(first_conv.permute(0, 3, 1, 2))
        first_conv2 = self.qnode_graph12(first_conv.permute(0, 3, 1, 2))
        first_conv = torch.mul(first_conv1, first_conv2).permute(0, 2, 3, 1)

        question_first_tree = first_conv

        question_third_tree = first_conv + q_rep + obj_qo

        # Perform scene attention by answer
        a_rep1 = self.answer_conv(a_rep.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        scene_attend = torch.matmul(a_rep1, scene_c[:, None])
        scene_attention_weights = masked_softmax(scene_attend, answer_mask[..., None])
        scene_ao = torch.einsum('bnao,bod->bnad', (scene_attention_weights, scene.permute(0, 2, 1)))

        obj_attend = torch.matmul(a_rep, obj_feas.permute(0, 2, 1)[:, None])
        obj_attention_weights = masked_softmax(obj_attend, answer_mask[..., None])
        obj_ao = torch.einsum('bnao,bod->bnad', (obj_attention_weights, obj_feas))

        answer_node = torch.cat([scene_ao, obj_ao, a_rep], -1)

        # answer_node = self.dropout(answer_node)

        ## answer first layer
        answer_diag = torch.diag_embed(
            answer_mask.view(answer_mask.shape[0] * answer_mask.shape[1], answer_mask.shape[2]))
        ans_diag = answer_diag.view(answer_mask.shape[0], answer_mask.shape[1], answer_mask.shape[2],
                                    answer_mask.shape[2]).float()
        first_graph = torch.matmul(answer_node, answer_node.permute(0, 1, 3, 2))
        first_mask = torch.matmul(answer_mask[..., None].float(), answer_mask[:, :, None, :].float())
        first_graph = masked_softmax(first_graph, first_mask, dim=2) + ans_diag

        first_conv = torch.matmul(first_graph, answer_node)
        first_conv1 = self.anode_graph11(first_conv.permute(0, 3, 1, 2))
        first_conv2 = self.anode_graph12(first_conv.permute(0, 3, 1, 2))
        first_conv = torch.mul(first_conv1, first_conv2).permute(0, 2, 3, 1)

        answer_first_tree = first_conv

        answer_third_tree = first_conv + a_rep + scene_ao

        # question and answer fusion
        question_third_tree1 = self.fusion_conv(question_third_tree.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        qa_tree_similarity = torch.matmul(question_third_tree1, a_rep.permute(0, 1, 3, 2))
        qa_tree_attention_weights = masked_softmax(qa_tree_similarity, question_mask[..., None], dim=2)
        attended_tree_q = torch.einsum('bnqa,bnqd->bnad', (qa_tree_attention_weights, question_third_tree))

        things_to_pool = torch.cat([attended_tree_q, answer_third_tree], -1)

        things_to_pool = self.reason_answer(things_to_pool) + attended_tree_q + answer_third_tree
        things_to_pool = torch.cat([things_to_pool, a_rep, obj_ao], -1)
        pooled_rep = replace_masked_values(things_to_pool, answer_mask[..., None], -1e7).max(2)[0]

        output_dict = {}
        output_dict['features_penult'] = pooled_rep
        logits = self.final_mlp(pooled_rep).squeeze(2)

        ###########################################

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict.update({"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                            'qa_attention_weights': qa_tree_attention_weights,
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