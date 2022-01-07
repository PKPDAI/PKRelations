from typing import Dict, List
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch as torch
from nervaluate import Evaluator
from pkrex.models.dataloaders import REX2ID, ID2REX
from pkrex.models.sampling import batch_index, collate_fn_padding
from pkrex.models.utils import get_ner_metrics, dpooler, bio_to_entity_tokens, assign_index_to_spans, \
    generate_all_possible_rels, filter_not_allowed_rels, possible_rels_to_entity_ids_format, get_entities_and_ctx_masks
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertModel
from sklearn.metrics import classification_report


class BertPKREX(pl.LightningModule):

    def __init__(self, config: Dict, id2tag: Dict[int, str], n_training_steps: int, pretrained_bert: BertModel = None):
        super(BertPKREX, self).__init__()
        # === 1. Set main variables ==== #

        self.run_name = config['run_name']
        self.weighted_loss = assign_property(inp_config=config, parameter_name='weighted_loss', alternative=False)
        self.scaling_dict = assign_property(inp_config=config, parameter_name='scaling_dict', alternative=None)
        self.out_path = config['output_dir']
        self.id2tag = id2tag
        self.nl = len(self.id2tag)
        if "PAD" in self.id2tag.values():
            self.nl -= 1
        self.n_training_steps = n_training_steps
        # === 2. Set main hyperparameters === #
        self.seq_len = config['max_length']
        self.lr = config['learning_rate']
        self.eps = config['eps']
        #    self.weight_decay = config['weight_decay']
        self.lr_warmup = assign_property(inp_config=config, parameter_name='lr_warmup', alternative=False)
        self.weight_decay = assign_property(inp_config=config, parameter_name='weight_decay', alternative=False)
        # === 3. Load model === #
        # self.model = load_model(model_path=config['base_model'], num_labels=self.nl)
        # NER
        if pretrained_bert:
            self.bert = pretrained_bert
        else:
            self.bert = AutoModel.from_pretrained(config['base_model'])
        self.dropout = torch.nn.Dropout(0.1)  # config['dropout_prob']
        self.ner_classifier = torch.nn.Linear(in_features=768,
                                              out_features=self.nl)
        # REX
        self.rel_classifier = torch.nn.Linear(768 * 4 + config['ctx_embedding_size'], len(set(REX2ID.values())))
        self.size_embeddings = torch.nn.Embedding(config['max_length'], config['ctx_embedding_size'])
        self.save_hyperparameters()

    def predict_relations(self, inp_sequence_rep, inp_ent_masks, inp_rex_masks, inp_ctx_masks,
                          inp_rel_tuples, inp_ctx_width):

        # 1. Entity pairs representations
        entity_reps = dpooler(entity_masks=inp_ent_masks, sentence_rep_batch=inp_sequence_rep)
        entity_pairs = batch_index(tensor=entity_reps, index=inp_rel_tuples)
        assert len(entity_pairs.shape) == 4  # bs x sample_rels x 2 (one per entity) x 768
        entity_pairs = torch.flatten(entity_pairs, start_dim=2, end_dim=3)  # convert to bs x sample_rels x 1536
        # 2. Context rep and add [CLS] if needed
        context_reps = dpooler(entity_masks=inp_ctx_masks, sentence_rep_batch=inp_sequence_rep)  # bs x sample_rels x
        # 768
        cls_tokens = inp_sequence_rep[:, 0, :]
        cls_tokens = cls_tokens.unsqueeze(dim=1).repeat(1, context_reps.shape[1], 1)
        context_reps = torch.cat([context_reps, cls_tokens], dim=2)

        # 3. Size embeddings
        size_reps = self.size_embeddings(inp_ctx_width)

        # 4. Flatten, concatenate
        entity_pairs = torch.flatten(entity_pairs, end_dim=1)  # (bs*sample_rels) x 1536
        context_reps = torch.flatten(context_reps, end_dim=1)
        context_reps = self.replace_zero_context(inp_context_tensor=context_reps)
        size_reps = torch.flatten(size_reps, end_dim=1)
        assert size_reps.shape[0] == context_reps.shape[0] == entity_pairs.shape[0] == inp_rex_masks.shape[0]
        all_rel_reps = torch.cat([entity_pairs, context_reps, size_reps], dim=1)
        all_rel_reps = all_rel_reps[inp_rex_masks]  # filter according to mask

        # 5. Pass through linear layer
        all_rel_reps = self.dropout(all_rel_reps)
        rex_logits = self.rel_classifier(all_rel_reps)
        return rex_logits

    def predict_entities(self, sequence_bert_output):
        sequence_bert_output = self.dropout(sequence_bert_output)
        return self.ner_classifier(sequence_bert_output)

    def forward(self, input_ids: torch.Tensor, attention_masks: torch.Tensor,
                entity_masks: torch.Tensor, rel_tuples: torch.Tensor,
                ctx_mask: torch.Tensor, ctx_width: torch.Tensor, rex_masks: torch.Tensor
                ):
        # 1. Pass through BERT
        h = self.bert(input_ids,
                      attention_mask=attention_masks)
        sequence_output = h[0]
        # 2. Predict NER
        ner_logits = self.predict_entities(sequence_bert_output=sequence_output)
        # 3. Predict REX
        rex_logits = None
        if rex_masks is not None and sum(rex_masks) != 0:  # if there are relations
            rex_logits = self.predict_relations(inp_sequence_rep=sequence_output, inp_ent_masks=entity_masks,
                                                inp_rex_masks=rex_masks, inp_ctx_masks=ctx_mask,
                                                inp_rel_tuples=rel_tuples, inp_ctx_width=ctx_width)
        return ner_logits, rex_logits

    def training_step(self, inp_batch, batch_nb):
        rex_masks = None
        if self.current_epoch > 0:  # wait for 2 epochs
            rex_masks = self.get_rex_masks(inp_rel_tuples=inp_batch['rel_tuples'])

        ner_logits, rex_logits = self(input_ids=inp_batch['input_ids'], attention_masks=inp_batch['attention_mask'],
                                      entity_masks=inp_batch['entity_masks'], rel_tuples=inp_batch['rel_tuples'],
                                      ctx_mask=inp_batch['ctx_mask'], ctx_width=inp_batch['ctx_len'],
                                      rex_masks=rex_masks)

        ner_loss = self.compute_ner_loss(ner_logits=ner_logits, ner_labels=inp_batch['labels'],
                                         inp_attention_masks=inp_batch['attention_mask'])
        rex_loss = torch.tensor(0.).to(self.device)

        if rex_logits is not None:
            labels_flat = torch.flatten(inp_batch['rel_labels'])[rex_masks]
            assert len(labels_flat.shape) == 1 and labels_flat.shape[0] == rex_logits.shape[0]
            rex_loss = self.compute_rex_loss(rex_logits=rex_logits, rex_labels=labels_flat)
        # print(rex_loss.item())
        loss = ner_loss + rex_loss

        return {'loss': loss, 'rex_loss': rex_loss}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_rex_loss = torch.stack([x['rex_loss'] for x in outputs]).mean()
        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_rex_loss', train_rex_loss, prog_bar=True)

    def get_rex_masks(self, inp_rel_tuples: torch.Tensor):
        rex_masks = None
        # if inp_rel_tuples.dim() < 2:
        #     print(inp_rel_tuples)
        #     return rex_masks
        try:
            if inp_rel_tuples.nelement() != 0 and inp_rel_tuples[1].nelement() != 0:
                rex_masks = ~torch.all(torch.eq(torch.flatten(inp_rel_tuples, end_dim=1),
                                                torch.tensor([0, 0], dtype=torch.int64).to(self.device)),
                                       dim=1)
        except:
            pass

        return rex_masks

    def validation_step(self, val_batch, batch_nb):

        # 1. BERT pass
        h = self.bert(val_batch['input_ids'], attention_mask=val_batch['attention_mask'])[0]
        # 2. NER prediction
        ner_logits = self.predict_entities(sequence_bert_output=h)
        val_ner_loss = self.compute_ner_loss(ner_logits=ner_logits, ner_labels=val_batch['labels'],
                                             inp_attention_masks=val_batch['attention_mask'])
        ner_labels = val_batch['labels']
        all_results_ner, avg_results_ner, iob_predictions = self.compute_ner_f1s(
            predictions=ner_logits,
            labels=ner_labels,
            id2tag=self.id2tag)

        # 3. REX prediction
        # val_rex_loss = torch.tensor(0.).to(self.device)
        # rex_masks = self.get_rex_masks(inp_rel_tuples=val_batch['rel_tuples'])
        # 3.1 Create a prediction batch and the extra labels for ground truth entities that have not been predicted
        # and are part of a relation

        pred_batch, all_extra_labels = self.create_pred_rex_batch(ner_logits=ner_logits,
                                                                  ner_labels=ner_labels,
                                                                  ent_masks=val_batch['entity_masks'],
                                                                  rel_labels=val_batch['rel_labels'],
                                                                  rel_tuples=val_batch['rel_tuples'],
                                                                  )

        rex_masks = None
        if self.current_epoch > -1:  # wait for 2 epochs
            # print(pred_batch['rel_tuples'].shape)
            # if pred_batch['rel_tuples'].dim() < 2:
            #     a = 1
            rex_masks = self.get_rex_masks(inp_rel_tuples=pred_batch['rel_tuples'])

        rex_logits = None
        if rex_masks is not None and sum(rex_masks) != 0:
            rex_logits = self.predict_relations(inp_sequence_rep=h, inp_ent_masks=pred_batch['entity_masks'],
                                                inp_rex_masks=rex_masks, inp_ctx_masks=pred_batch['ctx_mask'],
                                                inp_rel_tuples=pred_batch['rel_tuples'],
                                                inp_ctx_width=pred_batch['ctx_len'])
        true_rex_labels = [y for x in all_extra_labels for y in x]
        true_rex_predictions = [REX2ID["NO_RELATION"] for _ in true_rex_labels]
        if rex_logits is not None:
            labels_flat = torch.flatten(pred_batch['rel_labels'])[rex_masks]
            rex_preds = rex_logits.argmax(dim=1)
            assert labels_flat.shape == rex_preds.shape
            true_rex_labels += labels_flat.tolist()
            true_rex_predictions += rex_preds.tolist()

        return {'val_loss': val_ner_loss,
                'val_micro_strict': avg_results_ner['micro']['strict'],
                'val_macro_strict': avg_results_ner['macro']['strict'],
                'val_pk_strict': all_results_ner['PK']['strict']['f1'],
                'val_value_strict': all_results_ner['VALUE']['strict']['f1'],
                'val_units_strict': all_results_ner['UNITS']['strict']['f1'],
                'val_range_strict': all_results_ner['RANGE']['strict']['f1'],
                'val_compare_strict': all_results_ner['COMPARE']['strict']['f1'],
                'true_rex_labels': true_rex_labels,
                'true_rex_preds': true_rex_predictions
                }

    @staticmethod
    def compute_rex_metrics(inp_labels: List[int], inp_predictions: List[int]):

        inp_labels = [ID2REX[x] for x in inp_labels]
        inp_predictions = [ID2REX[x] for x in inp_predictions]
        c_report = classification_report(y_true=inp_labels, y_pred=inp_predictions, output_dict=True,
                                         zero_division=0)
        return c_report

    @staticmethod
    def get_f1_from_cr(inp_rex_mentrics, inp_key):
        if inp_key in inp_rex_mentrics:
            return inp_rex_mentrics[inp_key]['f1-score']
        else:
            return None

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_micro_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_micro_strict")
        val_macro_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_macro_strict")
        val_pk_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_pk_strict")
        val_value_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_value_strict")
        val_units_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_units_strict")
        val_range_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_range_strict")
        val_compare_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_compare_strict")

        true_rex_preds = [p for x in outputs for p in x['true_rex_preds']]
        true_rex_labels = [p for x in outputs for p in x['true_rex_labels']]
        rex_metrics = self.compute_rex_metrics(inp_labels=true_rex_labels, inp_predictions=true_rex_preds)

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_micro_strict', val_micro_strict, prog_bar=True)
        self.log('val_macro_strict', val_macro_strict, prog_bar=True)
        self.log('val_pk_value_strict', (val_pk_strict + val_value_strict) / 2, prog_bar=True)
        self.log('val_pk_strict', val_pk_strict, prog_bar=True)
        self.log('val_value_strict', val_value_strict, prog_bar=True)
        self.log('val_units_strict', val_units_strict, prog_bar=True)
        self.log('val_range_strict', val_range_strict, prog_bar=True)
        self.log('val_compare_strict', val_compare_strict, prog_bar=True)
        self.log('cval_f1', self.get_f1_from_cr(inp_rex_mentrics=rex_metrics, inp_key='C_VAL'), prog_bar=True)
        self.log('dval_f1', self.get_f1_from_cr(inp_rex_mentrics=rex_metrics, inp_key='D_VAL'), prog_bar=True)
        self.log('related_f1', self.get_f1_from_cr(inp_rex_mentrics=rex_metrics, inp_key='RELATED'), prog_bar=True)
        self.log('norel_f1', self.get_f1_from_cr(inp_rex_mentrics=rex_metrics, inp_key='NO_RELATION'), prog_bar=True)

    def create_pred_rex_batch(self, ner_logits, ner_labels, ent_masks, rel_labels, rel_tuples):
        ner_predictions = ner_logits.argmax(dim=2)
        iob_predictions = [
            [self.id2tag[tok_pred] if tok_lab != -100 else 'O' for tok_pred, tok_lab in zip(sentence_pred,
                                                                                            sentence_lab)]
            for sentence_pred, sentence_lab in zip(ner_predictions.tolist(), ner_labels.tolist())
        ]
        all_extra_labels, pred_batch = [], []

        for i, iobs in enumerate(iob_predictions):
            entity_tokens = assign_index_to_spans(bio_to_entity_tokens(inp_bio_seq=iobs))
            candidate_rels = generate_all_possible_rels(inp_entities=entity_tokens)
            candidate_rels = filter_not_allowed_rels(inp_possible_rels=candidate_rels)
            candidate_rels_ent_ids = possible_rels_to_entity_ids_format(inp_entities=entity_tokens,
                                                                        inp_rels=candidate_rels)
            pred_ent_masks, pred_ctx_masks, pred_ctx_lengths = get_entities_and_ctx_masks(inp_entities=entity_tokens,
                                                                                          inp_rels=candidate_rels,
                                                                                          max_len=self.seq_len)
            extra_labels, labels_to_predict = self.associate_predicted_rex_labels(
                correct_entity_masks=ent_masks[i],
                predicted_entity_masks=pred_ent_masks,
                correct_rel_tuples=rel_tuples[i],
                predicted_rel_tuples=candidate_rels_ent_ids,
                correct_rel_labels=rel_labels[i],
            )
            tmp_batch_sample = dict(
                entity_masks=torch.tensor(pred_ent_masks).to(self.device),
                rel_tuples=torch.tensor(candidate_rels_ent_ids).to(self.device),
                ctx_mask=torch.tensor(pred_ctx_masks).to(self.device),
                rel_labels=torch.tensor(labels_to_predict).to(self.device),
                ctx_len=torch.tensor(pred_ctx_lengths).to(self.device)
            )
            pred_batch.append(tmp_batch_sample)

            all_extra_labels.append(extra_labels)

        # pad pred batch:
        pred_batch = collate_fn_padding(batch=pred_batch)

        return pred_batch, all_extra_labels

    def associate_predicted_rex_labels(self, correct_entity_masks, predicted_entity_masks,
                                       correct_rel_tuples, predicted_rel_tuples, correct_rel_labels):

        extra_labels, labels_to_predict = [], []
        tmp_masks = None
        if correct_rel_tuples.shape[0] != 0:
            # correct_rel_tuples[1].nelement() != 0:
            tmp_masks = ~torch.all(
                torch.eq(correct_rel_tuples, torch.tensor([0, 0], dtype=torch.int64).to(self.device)), dim=1)
            if sum(tmp_masks) == 0:
                tmp_masks = None
        if len(predicted_rel_tuples) == 0:
            if tmp_masks is not None:
                # Here we don't have pred entities but we have ground entities
                extra_labels = correct_rel_labels[tmp_masks].tolist()
            return extra_labels, labels_to_predict  # we also jump here when we don't have either of them
        else:
            if tmp_masks is None or sum(tmp_masks) == 0:
                labels_to_predict = [REX2ID["NO_RELATION"] for _ in predicted_rel_tuples]
                # Here we have pred entities but not ground truth
                return extra_labels, labels_to_predict
            else:
                # Here we have pred and ground entities
                assert correct_entity_masks.shape[1] == len(predicted_entity_masks[0])

                pred_tup_masks = [predicted_entity_masks[ent1idx] + predicted_entity_masks[ent2idx] for ent1idx, ent2idx
                                  in predicted_rel_tuples]
                ground_tup_masks = [correct_entity_masks[ent1idx].tolist() + correct_entity_masks[ent2idx].tolist()
                                    for ent1idx, ent2idx in correct_rel_tuples[tmp_masks].tolist()]

                correct_rel_labels = correct_rel_labels[tmp_masks]

                for pred_ent_pair_mask in pred_tup_masks:
                    if pred_ent_pair_mask in ground_tup_masks:
                        # pick label
                        idx = [i for i, x in enumerate(ground_tup_masks) if pred_ent_pair_mask == x]
                        if len(idx) != 1:
                            raise ValueError
                        tmp_label = correct_rel_labels[idx[0]].item()
                        labels_to_predict.append(tmp_label)
                    else:
                        labels_to_predict.append(REX2ID["NO_RELATION"])
                for i, correct_ent_pair_mask in enumerate(ground_tup_masks):
                    if correct_ent_pair_mask not in pred_tup_masks:  # missed entities
                        extra_labels.append(correct_rel_labels[i].item())

                assert len(labels_to_predict) == len(predicted_rel_tuples)
                return extra_labels, labels_to_predict

    def configure_optimizers(self):
        if self.weight_decay:
            optimizer = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=self.eps,
                              correct_bias=False, weight_decay=self.weight_decay)
        else:
            optimizer = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=self.eps,
                              correct_bias=False)

        if self.lr_warmup:
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=int(
                                                            round(self.n_training_steps * self.lr_warmup)
                                                        ),
                                                        num_training_steps=self.n_training_steps)
            return [optimizer], [scheduler]
        return optimizer

    def compute_ner_loss(self, ner_logits: torch.Tensor, ner_labels: torch.Tensor, inp_attention_masks: torch.Tensor):
        """
        Function adapted from https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#
        BertForTokenClassification
        @param ner_logits: Tensor with batch_size * seq_length * n_labels
        @param ner_labels: Tensor with batch_size * seq_length (list of integers with label id)
        @param inp_attention_masks: attention masks of the input sequence
        @return: Cross Entropy loss
        """
        if self.weighted_loss:
            weights_list = torch.tensor([self.scaling_dict[self.id2tag[i]] for i in range(0, self.nl)]).to(self.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights_list)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        active_loss = inp_attention_masks.view(-1) == 1
        active_logits = ner_logits.view(-1, self.nl)
        active_labels = torch.where(
            active_loss, ner_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
        )
        return loss_fct(active_logits, active_labels)

    @staticmethod
    def compute_rex_loss(rex_logits: torch.Tensor, rex_labels: torch.Tensor):
        loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(rex_logits, rex_labels)

    @staticmethod
    def compute_ner_f1s(predictions: torch.Tensor, labels: torch.Tensor, id2tag: Dict[int, str]):
        """
        @param predictions: Input tensor resulting from the softmax layer; batch_size * sequence_length * n_classes
        @param labels: Sequence length; batch_size * sequence_length * n_classes
        @param id2tag: Dictionary mapping label ids to BIO/BILOU schema
        @return: Strict and Partial F1 scores per entity type and Strict and Partial F1 scores micro and macro averaged
        """
        # Remove labels with -100
        predictions = predictions.argmax(dim=2)
        assert predictions.shape == labels.shape

        true_predictions = [
            [id2tag[token_prediction] for (token_prediction, token_label) in zip(sentence_pred, sentence_lab) if
             token_label != -100]
            for sentence_pred, sentence_lab in zip(predictions.tolist(), labels.tolist())
        ]

        true_labels = [
            [id2tag[token_label] for (token_prediction, token_label) in zip(sentence_pred, sentence_lab) if
             token_label != -100]
            for sentence_pred, sentence_lab in zip(predictions.tolist(), labels.tolist())
        ]
        entity_types = list(set([x.split("-")[1] for x in id2tag.values() if "-" in x]))
        evaluator = Evaluator(true_labels, true_predictions, tags=entity_types, loader="list")
        _, results_agg = evaluator.evaluate()
        all_results, avg_results = get_avg_ner_metrics(inp_results_agg=results_agg, inp_entity_types=entity_types)

        return all_results, avg_results, true_predictions

    @staticmethod
    def compute_mean_val(val_outputs: List[Dict], desired_field: str):
        all_valid_vals = [x[desired_field] for x in val_outputs if x[desired_field] is not None]
        if len(all_valid_vals) != 0:
            return sum(all_valid_vals) / len(all_valid_vals)
        return None

    def replace_zero_context(self, inp_context_tensor):
        """Replaces context vectors for relations that had no tokens in-between entities"""
        assert len(inp_context_tensor.shape) == 2
        z_vecs = torch.zeros(inp_context_tensor.shape[0], inp_context_tensor.shape[1]).to(self.device)
        return torch.where(inp_context_tensor < -1e29, z_vecs, inp_context_tensor)


def get_avg_ner_metrics(inp_results_agg, inp_entity_types):
    all_results_dict = dict()
    for entyp in inp_entity_types:
        all_results_dict[entyp] = dict(partial=dict(), strict=dict())
        assert inp_results_agg[entyp]['partial']['possible'] == inp_results_agg[entyp]['strict']['possible']
        all_results_dict[entyp]['support'] = inp_results_agg[entyp]['strict']['possible']
        all_results_dict[entyp]['partial']['precision'] = None
        all_results_dict[entyp]['partial']['recall'] = None
        all_results_dict[entyp]['partial']['f1'] = None
        all_results_dict[entyp]['strict']['precision'] = None
        all_results_dict[entyp]['strict']['recall'] = None
        all_results_dict[entyp]['strict']['f1'] = None
        if all_results_dict[entyp]['support'] > 0:
            p, r, f1 = get_ner_metrics(inp_results_agg[entyp]['partial'])
            all_results_dict[entyp]['partial']['precision'] = p
            all_results_dict[entyp]['partial']['recall'] = r
            all_results_dict[entyp]['partial']['f1'] = f1
            p, r, f1 = get_ner_metrics(inp_results_agg[entyp]['strict'])
            all_results_dict[entyp]['strict']['precision'] = p
            all_results_dict[entyp]['strict']['recall'] = r
            all_results_dict[entyp]['strict']['f1'] = f1

    avg_results = get_micro_macro_f1s(all_results_dict)
    return all_results_dict, avg_results


def get_micro_macro_f1s(inp_results_dict: Dict) -> Dict:
    out_results = dict(micro=dict(strict=None, partial=None), macro=dict(strict=None, partial=None))
    # Macro avg
    macro_strict_f1s = [v['strict']['f1'] for v in inp_results_dict.values() if v['support'] > 0]
    macro_partial_f1s = [v['partial']['f1'] for v in inp_results_dict.values() if v['support'] > 0]
    out_results['macro']['strict'] = 0.
    out_results['macro']['partial'] = 0.
    if macro_strict_f1s:
        out_results['macro']['strict'] = sum(macro_strict_f1s) / len(macro_strict_f1s)
    if macro_partial_f1s:
        out_results['macro']['partial'] = sum(macro_partial_f1s) / len(macro_partial_f1s)
    # Micro avg
    micro_strict_f1, micro_partial_f1 = 0., 0.
    if inp_results_dict.values() and sum([x['support'] for x in inp_results_dict.values()]):
        total_support = sum([x['support'] for x in inp_results_dict.values()])
        micro_strict_f1 = sum([v['strict']['f1'] * (v['support'] / total_support) for v in inp_results_dict.values()
                               if v['strict']['f1'] is not None])
        micro_partial_f1 = sum([v['strict']['f1'] * (v['support'] / total_support) for v in inp_results_dict.values()
                                if v['strict']['f1'] is not None])
    out_results['micro']['strict'] = micro_strict_f1
    out_results['micro']['partial'] = micro_partial_f1
    return out_results


def assign_property(inp_config: Dict, parameter_name: str, alternative):
    """Assigns property if exists in dictionary, otherwise returns alternative"""
    if parameter_name in inp_config.keys():
        return inp_config[parameter_name]
    return alternative


def load_pretrained_model(model_checkpoint_path, gpu):
    device = 'cpu'
    if gpu:
        device = 'cuda'

    return BertPKREX.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path,
        map_location=torch.device(device),

    )
