"""
Dataloaders for VCR
"""
import json
import os
import base64
import csv
import sys
import numpy as np
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask
from torch.utils.data import Dataset
from dataloaders.box_utils import load_image, resize_image, to_tensor_and_normalize
from dataloaders.mask_utils import make_mask
from dataloaders.bert_field import BertField
import h5py
import pickle
from copy import deepcopy
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR

csv.field_size_limit(sys.maxsize)

GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']


# Here's an example jsonl
# {
# "movie": "3015_CHARLIE_ST_CLOUD",
# "objects": ["person", "person", "person", "car"],
# "interesting_scores": [0],
# "answer_likelihood": "possible",
# "img_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.jpg",
# "metadata_fn": "lsmdc_3015_CHARLIE_ST_CLOUD/3015_CHARLIE_ST_CLOUD_00.23.57.935-00.24.00.783@0.json",
# "answer_orig": "No she does not",
# "question_orig": "Does 3 feel comfortable?",
# "rationale_orig": "She is standing with her arms crossed and looks disturbed",
# "question": ["Does", [2], "feel", "comfortable", "?"],
# "answer_match_iter": [3, 0, 2, 1],
# "answer_sources": [3287, 0, 10184, 2260],
# "answer_choices": [
#     ["Yes", "because", "the", "person", "sitting", "next", "to", "her", "is", "smiling", "."],
#     ["No", "she", "does", "not", "."],
#     ["Yes", ",", "she", "is", "wearing", "something", "with", "thin", "straps", "."],
#     ["Yes", ",", "she", "is", "cold", "."]],
# "answer_label": 1,
# "rationale_choices": [
#     ["There", "is", "snow", "on", "the", "ground", ",", "and",
#         "she", "is", "wearing", "a", "coat", "and", "hate", "."],
#     ["She", "is", "standing", "with", "her", "arms", "crossed", "and", "looks", "disturbed", "."],
#     ["She", "is", "sitting", "very", "rigidly", "and", "tensely", "on", "the", "edge", "of", "the",
#         "bed", ".", "her", "posture", "is", "not", "relaxed", "and", "her", "face", "looks", "serious", "."],
#     [[2], "is", "laying", "in", "bed", "but", "not", "sleeping", ".",
#         "she", "looks", "sad", "and", "is", "curled", "into", "a", "ball", "."]],
# "rationale_sources": [1921, 0, 9750, 25743],
# "rationale_match_iter": [3, 0, 2, 1],
# "rationale_label": 1,
# "img_id": "train-0",
# "question_number": 0,
# "annot_id": "train-0",
# "match_fold": "train-0",
# "match_index": 0,
# }

def _fix_tokenization(tokenized_sent, bert_embs, old_det_to_new_ind, obj_to_type, token_indexers, pad_ind=-1):
    """
    Turn a detection list into what we want: some text, as well as some tags.
    :param tokenized_sent: Tokenized sentence with detections collapsed to a list.
    :param old_det_to_new_ind: Mapping of the old ID -> new ID (which will be used as the tag)
    :param obj_to_type: [person, person, pottedplant] indexed by the old labels
    :return: tokenized sentence
    """

    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                new_ind = old_det_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("Oh no, the new index is negative! that means it's invalid. {} {}".format(
                        tokenized_sent, old_det_to_new_ind
                    ))
                text_to_use = GENDER_NEUTRAL_NAMES[
                    new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            new_tokenization_with_tags.append((tok, pad_ind))

    text_field = BertField([Token(x[0]) for x in new_tokenization_with_tags],
                           bert_embs,
                           padding_value=0)
    tags = SequenceLabelField([x[1] for x in new_tokenization_with_tags], text_field)
    return text_field, tags


def _my_fix_tokenization(tokenized_sent, bert_embs, old_det_to_new_ind, obj_to_type, non_tag_old_det_to_new_ind,
                         non_tag_det_info, token_indexers, pad_ind=-1):
    """
    Turn a detection list into what we want: some text, as well as some tags.
    :param tokenized_sent: Tokenized sentence with detections collapsed to a list.
    :param old_det_to_new_ind: Mapping of the old ID -> new ID (which will be used as the tag)
    :param obj_to_type: [person, person, pottedplant] indexed by the old labels
    :return: tokenized sentence
    """

    new_tokenization_with_tags = []
    for tok_idx, tok in enumerate(tokenized_sent):
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                new_ind = old_det_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("Oh no, the new index is negative! that means it's invalid. {} {}".format(
                        tokenized_sent, old_det_to_new_ind
                    ))
                text_to_use = GENDER_NEUTRAL_NAMES[
                    new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            # check if there is a non-tag match available
            if tok_idx in non_tag_det_info:
                old_idx = non_tag_det_info[tok_idx][0]
                new_idx = non_tag_old_det_to_new_ind[old_idx]
                new_tokenization_with_tags.append((tok, new_idx))
            else:
                new_tokenization_with_tags.append((tok, pad_ind))

    text_field = BertField([Token(x[0]) for x in new_tokenization_with_tags],
                           bert_embs,
                           padding_value=0)
    tags = SequenceLabelField([x[1] for x in new_tokenization_with_tags], text_field)
    return text_field, tags


class VCR(Dataset):
    def __init__(self, split, tasks=('0', '1', '2'), only_use_relevant_dets=True, add_image_as_a_box=True, embs_to_load='bert_da',
                 conditioned_answer_choice=0, logits_qr2a=False, features_penult=False, features_last=False):
        """

        :param split: train, val, or test
        :param mode: answer or rationale
        :param only_use_relevant_dets: True, if we will only use the detections mentioned in the question and answer.
                                       False, if we should use all detections.
        :param add_image_as_a_box:     True to add the image in as an additional 'detection'. It'll go first in the list
                                       of objects.
        :param embs_to_load: Which precomputed embeddings to load.
        :param conditioned_answer_choice: If you're in test mode, the answer labels aren't provided, which could be
                                          a problem for the QA->R task. Pass in 'conditioned_answer_choice=i'
                                          to always condition on the i-th answer.
        """
        self.split = split
        # self.mode = mode
        self.tasks = tasks
        self.only_use_relevant_dets = only_use_relevant_dets
        print("Only relevant dets" if only_use_relevant_dets else "Using all detections", flush=True)

        self.add_image_as_a_box = add_image_as_a_box
        self.conditioned_answer_choice = conditioned_answer_choice

        with open(os.path.join(VCR_ANNOTS_DIR, split, '{}.jsonl'.format(split)), 'r') as f:
            self.items = [json.loads(s) for s in f]

        if split not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val.")

        for task in self.tasks:
            if task not in ('0', '1', '2', '3', '4'):
                raise ValueError(f"Not such task {task}!")
        # if mode not in ('answer', 'rationale'):
        #     raise ValueError("split must be answer or rationale")

        self.token_indexers = {'elmo': ELMoTokenCharactersIndexer()}
        self.vocab = Vocabulary()

        with open(os.path.join(os.path.dirname(VCR_ANNOTS_DIR), 'dataloaders', 'cocoontology.json'), 'r') as f:
            coco = json.load(f)
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

        self.embs_to_load = embs_to_load
        if '0' in self.tasks:
            self.h5fn_0 = os.path.join(VCR_ANNOTS_DIR, split, f'bert_qr2a_answer_{self.split}.h5')
            print("Loading embeddings from {}".format(self.h5fn_0), flush=True)
        if '1' in self.tasks:
            self.h5fn_1 = os.path.join(VCR_ANNOTS_DIR, split, f'{self.embs_to_load}_answer_{self.split}.h5')
            print("Loading embeddings from {}".format(self.h5fn_1), flush=True)
        if '2' in self.tasks:
            self.h5fn_2 = os.path.join(VCR_ANNOTS_DIR, split, f'{self.embs_to_load}_rationale_{self.split}.h5')
            print("Loading embeddings from {}".format(self.h5fn_2), flush=True)

        # load tag related stuff
        print('loading tag image features')
        self.tag_feature_path = os.path.join(VCR_ANNOTS_DIR, split, f'attribute_features_{self.split}.h5')

        # load non-tag related stuff
        print('loading non tag image features')
        self.non_tag_feature_path = os.path.join(VCR_ANNOTS_DIR, split, f'new_tag_features_{self.split}.h5')

        # load inverted index for non-tag detections. word idx -> obj indices这个地方关注一下对我们的QR2A是不是有影响
        annotid2detidx = None
        non_tag_question_annotid2detidx = None
        non_tag_answer_annotid2detidx = None
        non_tag_rationale_annotid2detidx = None

        non_tag_question_annotid2detidx_path = VCR_ANNOTS_DIR + f'/{split}_pickles_first_sense_match/question_annotid2detidx.pkl'
        non_tag_answer_annotid2detidx_path = VCR_ANNOTS_DIR + f'/{split}_pickles_first_sense_match/answer_annotid2detidx.pkl'
        non_tag_rationale_annotid2detidx_path = VCR_ANNOTS_DIR + f'/{split}_pickles_first_sense_match/rationale_annotid2detidx.pkl'

        with open(non_tag_question_annotid2detidx_path, 'rb') as f:
            non_tag_question_annotid2detidx = pickle.load(f)
        with open(non_tag_answer_annotid2detidx_path, 'rb') as f:
            non_tag_answer_annotid2detidx = pickle.load(f)
        with open(non_tag_rationale_annotid2detidx_path, 'rb') as f:
            non_tag_rationale_annotid2detidx = pickle.load(f)
        self.non_tag_question_annotid2detidx = non_tag_question_annotid2detidx
        self.non_tag_answer_annotid2detidx = non_tag_answer_annotid2detidx
        self.non_tag_rationale_annotid2detidx = non_tag_rationale_annotid2detidx

        # 如果QR2A是预训练好的，加载QR2A的数据
        self.features_penult = features_penult
        self.features_last = features_last
        self.logits_qr2a = logits_qr2a
        # self.logits_qr2a = np.zeros((len(self), 4))
        self.qr2a_h5 = os.path.join(VCR_ANNOTS_DIR, split, f'tab_qr2a_{self.split}.h5')

    @property
    def is_train(self):
        return self.split == 'train'

    @classmethod
    def splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset"""
        kwargs_copy = {x: y for x, y in kwargs.items()}
        # if 'mode' not in kwargs:
        #     kwargs_copy['mode'] = 'answer'
        train = cls(split='train', **kwargs_copy)
        val = cls(split='val', **kwargs_copy)
        # test = cls(split='test', **kwargs_copy)
        return train, val

    @classmethod
    def eval_splits(cls, **kwargs):
        """ Helper method to generate splits of the dataset. Use this for testing, because it will
            condition on everything."""
        for forbidden_key in ['mode', 'split', 'conditioned_answer_choice']:
            if forbidden_key in kwargs:
                raise ValueError(f"don't supply {forbidden_key} to eval_splits()")

        stuff_to_return = [cls(split='test', tasks=('1', '2'), conditioned_answer_choice=i, **kwargs) for i in range(4)]
        # stuff_to_return = [cls(split='test', mode='answer', **kwargs)] + [
        #     cls(split='test', mode='rationale', conditioned_answer_choice=i, **kwargs) for i in range(4)]
        return tuple(stuff_to_return)

    def __len__(self):
        return len(self.items)

    def _get_non_tag_det_to_use(self, non_tag_question_det_info, non_tag_answer_det_info, num_non_tag_boxes):
        dets2use = np.zeros(num_non_tag_boxes, dtype=bool)
        # question
        for obj_indices in list(non_tag_question_det_info.values()):
            dets2use[obj_indices[
                0]] = True  # we only use the 0 th detection index because we have averaged same class features and put it at 0 th idx

        # answer
        for sen in non_tag_answer_det_info:
            for obj_indices in list(sen.values()):
                dets2use[obj_indices[0]] = True

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros((num_non_tag_boxes), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)
        # old_det_to_new_ind = old_det_to_new_ind.tolist()

        return dets2use, old_det_to_new_ind

    def _get_non_tag_det_to_use_qar(self, non_tag_question_det_info, non_tag_answer_det_info, non_tag_rationale_det_info, num_non_tag_boxes):
        dets2use = np.zeros(num_non_tag_boxes, dtype=bool)
        # question
        for obj_indices in list(non_tag_question_det_info.values()):
            dets2use[obj_indices[
                0]] = True  # we only use the 0 th detection index because we have averaged same class features and put it at 0 th idx

        # answer
        for sen in non_tag_answer_det_info:
            for obj_indices in list(sen.values()):
                dets2use[obj_indices[0]] = True

        # rationale
        for sen in non_tag_rationale_det_info:
            for obj_indices in list(sen.values()):
                dets2use[obj_indices[0]] = True

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros((num_non_tag_boxes), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)
        # old_det_to_new_ind = old_det_to_new_ind.tolist()

        return dets2use, old_det_to_new_ind

    def _get_dets_to_use(self, item):
        """
        We might want to use fewer detectiosn so lets do so.
        :param item:
        :param question:
        :param answer_choices:
        :return:
        """
        # Load questions and answers
        question = item['question']
        answer_choices = item['answer_choices']
        rationale_choices = item['rationale_choices']

        if self.only_use_relevant_dets:
            dets2use = np.zeros(len(item['objects']), dtype=bool)
            people = np.array([x == 'person' for x in item['objects']], dtype=bool)
            for sent in answer_choices + [question] + rationale_choices:
                for possibly_det_list in sent:
                    if isinstance(possibly_det_list, list):
                        for tag in possibly_det_list:
                            if tag >= 0 and tag < len(item['objects']):  # sanity check
                                dets2use[tag] = True
                    elif possibly_det_list.lower() in ('everyone', 'everyones'):
                        dets2use |= people
            if not dets2use.any():
                dets2use |= people
        else:
            dets2use = np.ones(len(item['objects']), dtype=bool)

        # we will use these detections
        dets2use = np.where(dets2use)[0]

        old_det_to_new_ind = np.zeros(len(item['objects']), dtype=np.int32) - 1
        old_det_to_new_ind[dets2use] = np.arange(dets2use.shape[0], dtype=np.int32)

        # If we add the image as an extra box then the 0th will be the image.
        if self.add_image_as_a_box:
            old_det_to_new_ind[dets2use] += 1
        # old_det_to_new_ind = old_det_to_new_ind.tolist()
        return dets2use, old_det_to_new_ind

    def __getitem__(self, index):
        # if self.split == 'test':
        #     raise ValueError("blind test mode not supported quite yet")
        item = deepcopy(self.items[index])
        image_id = int(item['img_id'].split('-')[-1])

        with h5py.File(self.tag_feature_path, 'r') as h5:
            tag_features = np.array(h5[str(image_id)]['features'], dtype=np.float32)
            tag_boxes = np.array(h5[str(image_id)]['boxes'], dtype=np.float32)
            tag_obj_indices = np.array(h5[str(image_id)]['obj_indices'], dtype=np.int)

        with h5py.File(self.non_tag_feature_path, 'r') as h5:
            non_tag_boxes = np.array(h5[str(image_id)]['boxes'], dtype=np.float32)
            non_tag_obj_indices = np.array(h5[str(image_id)]['obj_indices'], dtype=np.int)
            non_tag_features = np.array(h5[str(image_id)]['features'], dtype=np.float32)
        ###################################################################
        # Load questions and answers 由于计算量不大，我这里统一把三种query（question, question_answer. question+rationale）和answer choice，rationale choice都加载出来

        non_tag_question_annotid2detidx = self.non_tag_question_annotid2detidx[item['annot_id']]
        non_tag_answer_annotid2detidx = self.non_tag_answer_annotid2detidx[item['annot_id']]
        non_tag_rationale_annotid2detidx = self.non_tag_rationale_annotid2detidx[item['annot_id']]

        question_annotid2detidx_1 = non_tag_question_annotid2detidx
        conditioned_label = item['answer_label'] if self.split != 'test' else self.conditioned_answer_choice
        q_len = len(item['question'])
        question_annotid2detidx_2 = {}
        for k, v in non_tag_question_annotid2detidx.items():
            question_annotid2detidx_2[k] = v
        for k, v in non_tag_answer_annotid2detidx[conditioned_label].items():
            question_annotid2detidx_2[k + q_len] = v
        # question_annotid2detidx_0 = {}
        # for k, v in non_tag_question_annotid2detidx.items():
        #     question_annotid2detidx_0[k] = v
        # for k, v in non_tag_rationale_annotid2detidx[item['rationale_label']].items():
        #     question_annotid2detidx_0[k + q_len] = v
        answer_annotid2detidx = non_tag_answer_annotid2detidx
        rationale_annotid2detidx = non_tag_rationale_annotid2detidx

        # if self.mode == 'answer':
        #     question_annotid2detidx = non_tag_question_annotid2detidx
        #     answer_annotid2detidx = non_tag_answer_annotid2detidx
        # else:
        #     conditioned_label = item['answer_label'] if self.split != 'test' else self.conditioned_answer_choice
        #     q_len = len(item['question'])
        #     question_annotid2detidx = {}
        #     for k, v in non_tag_question_annotid2detidx.items():
        #         question_annotid2detidx[k] = v
        #     for k, v in non_tag_answer_annotid2detidx[conditioned_label].items():
        #         question_annotid2detidx[k + q_len] = v
        #     answer_annotid2detidx = non_tag_rationale_annotid2detidx

        # if self.mode == 'rationale':
        #     conditioned_label = item['answer_label'] if self.split != 'test' else self.conditioned_answer_choice
        #     item['question'] += item['answer_choices'][conditioned_label]

        # answer_choices = item['{}_choices'.format(self.mode)]
        dets2use, old_det_to_new_ind = self._get_dets_to_use(item)
        non_tag_dets2use, non_tag_old_det_to_new_ind = self._get_non_tag_det_to_use_qar(question_annotid2detidx_1,
                                                                                        answer_annotid2detidx,
                                                                                        rationale_annotid2detidx,
                                                                                        len(non_tag_boxes))

        if self.add_image_as_a_box:
            assert (len(dets2use) == np.max(old_det_to_new_ind))

        if self.add_image_as_a_box:
            non_tag_old_det_to_new_ind += 1

        # shift the non_tag detection idx, effectively as appending the non_tag detections to tag detections
        non_tag_old_det_to_new_ind[np.where(non_tag_old_det_to_new_ind)[0]] += len(dets2use)

        old_det_to_new_ind = old_det_to_new_ind.tolist()
        non_tag_old_det_to_new_ind = non_tag_old_det_to_new_ind.tolist()

        instance_dict = {}
        #####################################  Q2A part  ##############################
        if '1' in self.tasks:
            with h5py.File(self.h5fn_1, 'r') as h5:
                grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}

            if 'endingonly' not in self.embs_to_load:
                questions_tokenized, question_tags = zip(*[_my_fix_tokenization(
                    item['question'],
                    grp_items[f'ctx_answer{i}'],
                    old_det_to_new_ind,
                    item['objects'],
                    non_tag_old_det_to_new_ind,
                    question_annotid2detidx_1,
                    token_indexers=self.token_indexers,
                    pad_ind=0 if self.add_image_as_a_box else -1,
                ) for i in range(4)])
                instance_dict['question_1'] = ListField(questions_tokenized)
                instance_dict['question_tags_1'] = ListField(question_tags)

            answers_tokenized, answer_tags = zip(*[_my_fix_tokenization(
                answer,
                grp_items[f'answer_answer{i}'],
                old_det_to_new_ind,
                item['objects'],
                non_tag_old_det_to_new_ind,
                answer_annotid2detidx[i],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1,
            ) for i, answer in enumerate(item['answer_choices'])])

            instance_dict['answers_1'] = ListField(answers_tokenized)
            instance_dict['answer_tags_1'] = ListField(answer_tags)

        #####################################  QA2R part  ##############################
        if '2' in self.tasks:
            with h5py.File(self.h5fn_2, 'r') as h5:
                grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}
            conditioned_label = item['answer_label'] if self.split != 'test' else self.conditioned_answer_choice
            question_2 = item['question'] + item['answer_choices'][conditioned_label]
            condition_key = self.conditioned_answer_choice if self.split == "test" else ""
            if 'endingonly' not in self.embs_to_load:
                questions_tokenized, question_tags = zip(*[_my_fix_tokenization(
                    question_2,
                    grp_items[f'ctx_rationale{condition_key}{i}'],
                    old_det_to_new_ind,
                    item['objects'],
                    non_tag_old_det_to_new_ind,
                    question_annotid2detidx_2,
                    token_indexers=self.token_indexers,
                    pad_ind=0 if self.add_image_as_a_box else -1,
                ) for i in range(4)])
                instance_dict['question_2'] = ListField(questions_tokenized)
                instance_dict['question_tags_2'] = ListField(question_tags)

            answers_tokenized, answer_tags = zip(*[_my_fix_tokenization(
                answer,
                grp_items[f'answer_rationale{condition_key}{i}'],
                old_det_to_new_ind,
                item['objects'],
                non_tag_old_det_to_new_ind,
                rationale_annotid2detidx[i],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1,
            ) for i, answer in enumerate(item['rationale_choices'])])
            instance_dict['answers_2'] = ListField(answers_tokenized)
            instance_dict['answer_tags_2'] = ListField(answer_tags)

        #####################################  QR2A part  ##############################
        if '0' in self.tasks:
            question_annotid2detidx_0 = {}
            for k, v in non_tag_question_annotid2detidx.items():
                question_annotid2detidx_0[k] = v
            for k, v in non_tag_rationale_annotid2detidx[item['rationale_label']].items():
                question_annotid2detidx_0[k + q_len] = v

            with h5py.File(self.h5fn_0, 'r') as h5:
                grp_items = {k: np.array(v, dtype=np.float16) for k, v in h5[str(index)].items()}
            question_0 = item['question'] + item['rationale_choices'][item['rationale_label']]
            if 'endingonly' not in self.embs_to_load:
                questions_tokenized, question_tags = zip(*[_my_fix_tokenization(
                    question_0,
                    grp_items[f'ctx_answer{i}'],
                    old_det_to_new_ind,
                    item['objects'],
                    non_tag_old_det_to_new_ind,
                    question_annotid2detidx_0,
                    token_indexers=self.token_indexers,
                    pad_ind=0 if self.add_image_as_a_box else -1,
                ) for i in range(4)])
                instance_dict['question_0'] = ListField(questions_tokenized)
                instance_dict['question_tags_0'] = ListField(question_tags)

            answers_tokenized, answer_tags = zip(*[_my_fix_tokenization(
                answer,
                grp_items[f'answer_answer{i}'],
                old_det_to_new_ind,
                item['objects'],
                non_tag_old_det_to_new_ind,
                answer_annotid2detidx[i],
                token_indexers=self.token_indexers,
                pad_ind=0 if self.add_image_as_a_box else -1,
            ) for i, answer in enumerate(item['answer_choices'])])

            instance_dict['answers_0'] = ListField(answers_tokenized)
            instance_dict['answer_tags_0'] = ListField(answer_tags)

        if self.split != 'test':
            instance_dict['answer_label'] = LabelField(item['answer_label'], skip_indexing=True)
            instance_dict['rationale_label'] = LabelField(item['rationale_label'], skip_indexing=True)
            # instance_dict['index'] = LabelField(index, skip_indexing=True)
        instance_dict['metadata'] = MetadataField({'annot_id': item['annot_id'], 'ind': index, 'movie': item['movie'],
                                                   'img_fn': item['img_fn'],
                                                   'question_number': item['question_number']})

        ###################################################################
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.

        ###################################################################
        # Load boxes.
        with open(os.path.join(VCR_IMAGES_DIR, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        # Chop off the final dimension, that's the confidence
        tag_boxes = np.array(metadata['boxes'])[dets2use, :-1]
        if self.add_image_as_a_box:
            tag_boxes = np.row_stack(([1, 1, 700, 700], tag_boxes))  # here we just use dummy box for background
        non_tag_boxes = non_tag_boxes[non_tag_dets2use]
        boxes = np.concatenate((tag_boxes, non_tag_boxes))

        if self.add_image_as_a_box:
            dets2use = dets2use + 1
            dets2use = np.insert(dets2use, 0, 0)

        tag_det_features = tag_features[dets2use]
        non_tag_det_features = non_tag_features[non_tag_dets2use]
        det_features = np.concatenate((tag_det_features, non_tag_det_features))

        instance_dict['det_features'] = ArrayField(det_features, padding_value=0)
        assert (det_features.shape[0] == boxes.shape[0])

        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)

        if self.logits_qr2a or self.features_penult or self.features_last:
            with h5py.File(self.qr2a_h5, 'r') as h5:
                h5_i = h5[str(index)]
                answer_label = int(np.array(h5_i['answer_label'], dtype=np.int64))
                if self.logits_qr2a:
                    instance_dict['logits_qr2a'] = ArrayField(np.array(h5_i['label_logits'], dtype=np.float32), padding_value=-1)
                if self.features_penult:
                    instance_dict['features_penult_qr2a'] = ArrayField(np.array(h5_i['features_penult'], dtype=np.float32), padding_value=-1)
                if self.features_last:
                    instance_dict['features_last_qr2a'] = ArrayField(np.array(h5_i['features_last'], dtype=np.float32), padding_value=-1)
                # instance_dict['qr2a_probs'] = ArrayField(np.array(h5_i['label_probs'], dtype=np.float32), padding_value=-1)
                assert answer_label == item['answer_label']

        instance = Instance(instance_dict)
        instance.index_fields(self.vocab)
        return None, instance


def collate_fn(data, to_gpu=False):
    """Creates mini-batch tensors
    """
    images, instances = zip(*data)
    batch = Batch(instances)
    td = batch.as_tensor_dict()
    for i in range(3):
        if f'question_{i}' in td:
            td[f'question_mask_{i}'] = get_text_field_mask(td[f'question_{i}'], num_wrapping_dims=1)
            td[f'question_tags_{i}'][td[f'question_mask_{i}'] == 0] = -2  # Padding

    for i in range(3):
        if f'answers_{i}' in td:
            td[f'answer_mask_{i}'] = get_text_field_mask(td[f'answers_{i}'], num_wrapping_dims=1)
            td[f'answer_tags_{i}'][td[f'answer_mask_{i}'] == 0] = -2

    td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()

    return td


class VCRLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, batch_size=3, num_workers=6, num_gpus=3, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size * num_gpus,
            # shuffle=False,
            shuffle=data.is_train,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, to_gpu=False),
            drop_last=data.is_train,
            # drop_last=False,
            pin_memory=False,
            **kwargs,
        )
        return loader


def get_batch_tasks(batch, tasks):
    batch_tasks = {}
    for task in tasks:
        batch_tasks[task] = {'det_features': batch['det_features'], 'boxes': batch['boxes'], 'box_mask': batch['box_mask'], 'metadata': batch['metadata'],
                             'question': batch[f'question_{task}'],
                             'question_tags': batch[f'question_tags_{task}'],
                             'question_mask': batch[f'question_mask_{task}'],
                             'answers': batch[f'answers_{task}'], 'answer_tags': batch[f'answer_tags_{task}'],
                             'answer_mask': batch[f'answer_mask_{task}'],
                             }
        if task in ('0', '1', '3') and 'answer_label' in batch:
            batch_tasks[task]['label'] = batch['answer_label']
        if task in ('2', '4') and 'rationale_label' in batch:
            batch_tasks[task]['label'] = batch['rationale_label']
    return batch_tasks


def get_batch_tasks2(batch, tasks):
    batch_tasks = {'det_features': batch['det_features'], 'boxes': batch['boxes'], 'box_mask': batch['box_mask'], 'metadata': batch['metadata'],}

    if 'logits_qr2a' in batch:
        batch_tasks['logits_qr2a'] = batch['logits_qr2a']
    if 'features_penult_qr2a' in batch:
        batch_tasks['features_penult_qr2a'] = batch['features_penult_qr2a']
    if 'features_last_qr2a' in batch:
        batch_tasks['features_last_qr2a'] = batch['features_last_qr2a']
    for task in tasks:
        batch_tasks[f'input_dict_{task}'] = {
            'question': batch[f'question_{task}'], 'question_tags': batch[f'question_tags_{task}'], 'question_mask': batch[f'question_mask_{task}'],
            'answers': batch[f'answers_{task}'], 'answer_tags': batch[f'answer_tags_{task}'], 'answer_mask': batch[f'answer_mask_{task}'],
        }
        if task in ('0', '1', '3') and 'answer_label' in batch:
            batch_tasks[f'input_dict_{task}']['label'] = batch['answer_label']
        if task in ('2', '4') and 'rationale_label' in batch:
            batch_tasks[f'input_dict_{task}']['label'] = batch['rationale_label']
    return batch_tasks

# You could use this for debugging maybe
if __name__ == '__main__':
    train, val = VCR.splits(tasks=('1', '2'), logits_qr2a=True, features_penult=True, features_last=True)
    for i in range(len(train)):
        res = train[i]
        print("done with {}".format(i))
