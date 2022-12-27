import logging
import os
import sys
from datetime import datetime
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval
from trainer.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils.utils_qa import check_no_error, postprocess_qa_predictions
from data_loaders.data_loader import load_train_dataset, load_eval_dataset
from run_sparse_retrieval import run_sparse_retrieval

logger = logging.getLogger(__name__)

def run_mrc(
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        datasets: DatasetDict,
        tokenizer,
        model,
    ) -> NoReturn:

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    train_dataset = load_train_dataset(datasets, max_seq_length, data_args, tokenizer)
    eval_dataset = load_eval_dataset(datasets, max_seq_length, data_args, tokenizer)

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
            examples,
            features,
            predictions: Tuple[np.ndarray, np.ndarray],
            training_args: TrainingArguments,
        ) -> EvalPrediction:

        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        answer_column_name = "answers" if "answers" in datasets["validation"].column_names else datasets["validation"].column_names[2]
        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]}
            for ex in datasets["validation"]
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

        # True일 경우 : run passage retrieval
        if data_args.eval_retrieval:
            # Evaluate
            training_dir = training_args.output_dir
            eval_dir = os.path.join(training_dir, "eval")
            if not os.path.exists(eval_dir): os.makedirs(eval_dir)
            training_args.output_dir = eval_dir

            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            test_datasets = load_from_disk('./dataset/test_dataset/')

            # Predict            
            predict_dir = os.path.join(training_dir, "pred")
            os.makedirs(predict_dir)
            training_args.output_dir = predict_dir
            predict_datasets = run_sparse_retrieval(
                tokenizer.tokenize, test_datasets, training_args, data_args,
            )
            predict_dataset = load_eval_dataset(predict_datasets, max_seq_length, data_args, tokenizer)

            logger.info("*** Predict ***")
            predictions = trainer.predict(
                test_dataset=predict_dataset, test_examples=predict_datasets["validation"]
            )
            # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
            print(
                "No metric can be presented because there is no correct answer given. Job done!"
            )

    # Evaluation
    if training_args.do_eval:
        # Evaluate
        training_dir = training_args.output_dir
        eval_dir = os.path.join(training_dir, "eval")
        os.makedirs(eval_dir)
        training_args.output_dir = eval_dir

        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    
    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        # Predict            
        test_datasets = load_from_disk('./dataset/test_dataset/')
        training_dir = training_args.output_dir
        predict_dir = os.path.join(training_dir, "pred")
        os.makedirs(predict_dir)
        training_args.output_dir = predict_dir
        predict_datasets = run_sparse_retrieval(
            tokenizer.tokenize, test_datasets, training_args, data_args,
        )
        predict_dataset = load_eval_dataset(predict_datasets, max_seq_length, data_args, tokenizer)

        logger.info("*** Predict ***")
        predictions = trainer.predict(
            test_dataset=predict_dataset, test_examples=predict_datasets["validation"]
        )
        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )
