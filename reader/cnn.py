import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules import dropout
from torch.nn.modules.conv import Conv1d
from typing import Optional, Tuple, Union
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    QuestionAnsweringModelOutput,
    ROBERTA_START_DOCSTRING,
    ROBERTA_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    QuestionAnsweringModelOutput,
    DEBERTA_START_DOCSTRING,
    DEBERTA_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _CHECKPOINT_FOR_QA,
    _QA_EXPECTED_OUTPUT,
    _QA_EXPECTED_LOSS,
    _QA_TARGET_START_INDEX,
    _QA_TARGET_END_INDEX,
)
from transformers import AutoModel


@add_start_docstrings(
    """
    Roberta Model with a LSTM span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
    tranformer RobertaForQuestionAnswering class를 변형
    """,
    ROBERTA_START_DOCSTRING,
)
class Conv1DRobertaForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, pretrained_model_name_or_path, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        assert "roberta" in config.model_type.lower(), "Base model does not match with any Roberta variants"

        self.roberta = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=config, add_pooling_layer=False
        )

        self.hidden_dim = config.hidden_size

        self.qa_outputs_base = nn.Linear(in_features=self.hidden_dim, out_features=config.num_labels)
        self.qa_outputs_lstm = nn.Linear(in_features=self.hidden_dim * 2, out_features=config.num_labels)
        self.qa_outputs_cnn = nn.Linear(in_features=self.hidden_dim * 3, out_features=config.num_labels)

        self.conv1d_k1 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=384, kernel_size=1, padding=0)
        self.conv1d_k3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=384, kernel_size=3, padding=1)
        self.conv1d_k5 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=384, kernel_size=5, padding=2)
       
        self.lstm = nn.LSTM(
            input_size=384 * 3,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )

        self.relu = nn.ReLU()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        '''
        RobertaForQuestionAnswering에서 수정 시작
        원본 코드 : 
            sequence_output = outputs[0]
            logits = self.qa_outputs(sequence_output)
        1. roberta 결과 [b,L,h]를 가지고 1칸 cnn, 3칸 cnn, 5칸 cnn을 각각 통과
        2. cnn 결과값을 concat
        3. (선택) lstm 돌리기
        4. linear로 start와 end 예측
        '''
        
        # outputs[0] = [batch_size,max_sequence_length=384, hidden_dim=1024]
        sequence_output = outputs[0].permute(0, 2, 1)   #차원바꾸기 [batch_size,hidden_dim,max_sequence_length]
        
        #1. cnn 작업
        cnn_k1_output = self.relu(self.conv1d_k1(sequence_output))  # 1칸 cnn [b, h, 384 = max_seq_len]
        cnn_k3_output = self.relu(self.conv1d_k3(sequence_output))  # 3칸 cnn
        cnn_k5_output = self.relu(self.conv1d_k5(sequence_output))  # 5칸 cnn

        #2. cnn concat
        concat_cnn_output = torch.cat((cnn_k1_output, cnn_k3_output, cnn_k5_output), 1) #cnn concat [b, h*3 ,384]
        sequence_output= concat_cnn_output.permute(0, 2, 1)   #[b, 384, h*3]
        
        #3(선택). lstm 사용
        sequence_output, (h, c) = self.lstm(sequence_output) # [batch_size, 384, 2048] <-- ??

        #4. lstm start와 end 예측
        #logits = self.qa_outputs_lstm(sequence_output)   # [batch_size, 384, num_label=2]
        #4. base_code
        #logits = self.qa_outputs_base(sequence_output)
        #4. cnn
        logits = self.qa_outputs_cnn(sequence_output)
        
        '''
        RobertaForQuestionAnswering에서 수정 끝
        '''
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
# Copied from transformers.models.deberta.modeling_deberta.DebertaForQuestionAnswering with Deberta->DebertaV2
class Conv1DDebertaV2ForQuestionAnswering(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, pretrained_model_name_or_path, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=config
        )

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        self.hidden_dim = config.hidden_size

        self.qa_outputs_base = nn.Linear(in_features=self.hidden_dim, out_features=config.num_labels)
        self.qa_outputs_lstm = nn.Linear(in_features=self.hidden_dim * 2, out_features=config.num_labels)
        self.qa_outputs_cnn = nn.Linear(in_features=self.hidden_dim * 3, out_features=config.num_labels)

        self.conv1d_k1 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=384, kernel_size=1, padding=0)
        self.conv1d_k3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=384, kernel_size=3, padding=1)
        self.conv1d_k5 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=384, kernel_size=5, padding=2)
       
        self.lstm = nn.LSTM(
            input_size=384 * 3,
            hidden_size=self.hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )

        self.relu = nn.ReLU()

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        '''
        debertaForQuestionAnswering에서 수정 시작
        원본 코드 : 
            sequence_output = outputs[0]
            logits = self.qa_outputs(sequence_output)
        1. roberta 결과 [b,L,h]를 가지고 1칸 cnn, 3칸 cnn, 5칸 cnn을 각각 통과
        2. cnn 결과값을 concat
        3. (선택) lstm 돌리기
        4. linear로 start와 end 예측
        '''
        
        # outputs[0] = [batch_size,max_sequence_length=384, hidden_dim=1024]
        sequence_output = outputs[0].permute(0, 2, 1)   #차원바꾸기 [batch_size,hidden_dim,max_sequence_length]
        
        #1. cnn 작업
        cnn_k1_output = self.relu(self.conv1d_k1(sequence_output))  # 1칸 cnn [b, h, 384 = max_seq_len]
        cnn_k3_output = self.relu(self.conv1d_k3(sequence_output))  # 3칸 cnn
        cnn_k5_output = self.relu(self.conv1d_k5(sequence_output))  # 5칸 cnn

        #2. cnn concat
        concat_cnn_output = torch.cat((cnn_k1_output, cnn_k3_output, cnn_k5_output), 1) #cnn concat [b, h*3 ,384]
        sequence_output= concat_cnn_output.permute(0, 2, 1)   #[b, 384, h*3]
        
        #3(선택). lstm 사용
        sequence_output, (h, c) = self.lstm(sequence_output) # [batch_size, 384, 2048] <-- ??

        #4. lstm start와 end 예측
        logits = self.qa_outputs_lstm(sequence_output)   # [batch_size, 384, num_label=2]
        #4. base_code
        #logits = self.qa_outputs_base(sequence_output)
        #4. cnn
        #logits = self.qa_outputs_cnn(sequence_output)
        
        '''
        debertaForQuestionAnswering에서 수정 끝
        '''
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )