from typing import Dict, List
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch as torch
from nervaluate import Evaluator
from pkrex.models.utils import get_ner_metrics
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertModel


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
        if pretrained_bert:
            self.bert = pretrained_bert
        else:
            self.bert = AutoModel.from_pretrained(config['base_model'])
        self.dropout = torch.nn.Dropout(0.1)  # config['dropout_prob']
        self.ner_classifier = torch.nn.Linear(in_features=768,
                                              out_features=self.nl)
        self.save_hyperparameters()

    def forward(self, input_ids: torch.Tensor, attention_masks: torch.Tensor):
        """
        Adapted from https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html
        """
        outputs = self.bert(input_ids,
                            attention_mask=attention_masks)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        return self.ner_classifier(sequence_output)

    def training_step(self, inp_batch, batch_nb):

        batch_logits = self(input_ids=inp_batch['input_ids'], attention_masks=inp_batch['attention_mask'])

        loss = self.compute_ner_loss(ner_logits=batch_logits, ner_labels=inp_batch['labels'],
                                     inp_attention_masks=inp_batch['attention_mask'])

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', train_loss, prog_bar=True)

    def validation_step(self, val_batch, batch_nb):

        batch_logits = self(input_ids=val_batch['input_ids'], attention_masks=val_batch['attention_mask'])

        val_loss = self.compute_ner_loss(ner_logits=batch_logits, ner_labels=val_batch['labels'],
                                         inp_attention_masks=val_batch['attention_mask'])

        all_results, avg_results = self.compute_ner_f1s(
            predictions=batch_logits,
            labels=val_batch['labels'],
            id2tag=self.id2tag)

        return {'val_loss': val_loss,
                'val_micro_strict': avg_results['micro']['strict'],
                'val_macro_strict': avg_results['macro']['strict'],
                'val_pk_strict': all_results['PK']['strict']['f1'],
                'val_value_strict': all_results['VALUE']['strict']['f1'],
                'val_units_strict': all_results['UNITS']['strict']['f1'],
                'val_range_strict': all_results['RANGE']['strict']['f1'],
                'val_compare_strict': all_results['COMPARE']['strict']['f1']
                }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_micro_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_micro_strict")
        val_macro_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_macro_strict")
        val_pk_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_pk_strict")
        val_value_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_value_strict")
        val_units_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_units_strict")
        val_range_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_range_strict")
        val_compare_strict = self.compute_mean_val(val_outputs=outputs, desired_field="val_compare_strict")

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_micro_strict', val_micro_strict, prog_bar=True)
        self.log('val_macro_strict', val_macro_strict, prog_bar=True)
        self.log('val_pk_value_strict', (val_pk_strict + val_value_strict) / 2, prog_bar=True)
        self.log('val_pk_strict', val_pk_strict, prog_bar=True)
        self.log('val_value_strict', val_value_strict, prog_bar=True)
        self.log('val_units_strict', val_units_strict, prog_bar=True)
        self.log('val_range_strict', val_range_strict, prog_bar=True)
        self.log('val_compare_strict', val_compare_strict, prog_bar=True)

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

        return all_results, avg_results

    @staticmethod
    def compute_mean_val(val_outputs: List[Dict], desired_field: str):
        all_valid_vals = [x[desired_field] for x in val_outputs if x[desired_field] is not None]
        if len(all_valid_vals) != 0:
            return sum(all_valid_vals) / len(all_valid_vals)
        return None


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
    out_results['macro']['strict'] = sum(macro_strict_f1s) / len(macro_strict_f1s)
    out_results['macro']['partial'] = sum(macro_partial_f1s) / len(macro_partial_f1s)
    # Micro avg
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
