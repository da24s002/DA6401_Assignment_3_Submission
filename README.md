# DA6401_Assignment_3
Sequence to Sequence transliteration assignment for DA6401 course

Implementation of a Sequencee to Sequence models with encoder, decoder architecture with RNN, GRU and LSTM

In order to run the model you can run the following command :

runs with the best arguments found while experimenting with the Daskshina Dataset<br>
## python train.py 

arguments supported :

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | DA6401_Assignment_3 | WandB project name |
| `--wandb_entity` | None | WandB entity name |
| `--model_type` | attention | Type of model to use (basic or attention) |
| `--encoder_rnn_type` | LSTM | Type of RNN to use for encoder |
| `--decoder_rnn_type` | LSTM | Type of RNN to use for decoder |
| `--embedding_dim` | 64 | Dimension of embedding vectors |
| `--hidden_dim` | 512 | Dimension of hidden states in RNN |
| `--encoder_layers` | 3 | Number of layers in encoder RNN |
| `--decoder_layers` | 2 | Number of layers in decoder RNN |
| `--dropout` | 0.2 | Dropout rate |
| `--batch_size` | 16 | Batch size for training |
| `--epochs` | 10 | Number of epochs to train |
| `--learning_rate` | 0.0007625795353002294 | Learning rate for optimizer |
| `--clip` | 5 | Gradient clipping value |
| `--teacher_forcing_ratio` | 1 | Probability of using teacher forcing during training |
| `--decode_method` | greedy | Decoding method (greedy or beam search) |
| `--beam_width` | 5 | Beam width for beam search decoding |
| `--length_penalty_alpha` | 1.0 | Length penalty alpha for beam search (higher values favor longer sequences) |
| `--compare_decoding` | False | Compare both greedy and beam search decoding on test examples |
| `--language` | hi | Language code for Dakshina dataset |
| `--max_seq_len` | 50 | Maximum sequence length |
| `--enable_visualizations` | 0 | If you want to generate the plots, pass this argument as 1 |
| `--generate_test_predictions` | 0 | If you want to generate test predictions |
| `--enable_wandb` | 0 | If you want to enable wandb logs |

The final test accuracy is reported in console after all the epochs are run.

Please go through the wandb init workflow in your current directory before running the code, as the code is configured to automatically log the run into wandb.

Run the command:
## wandb init

<br>
then provide the options asked, (note that the default project name is provided in the key value pair : "wandb_project": "DA6401_Assignment_3", you can ## change it while providing command line argument or directly in the dictionary, if you use a different wandb project while initialization)

==========================================================================================

Running a wandb sweep,

eg:

## wandb sweep config.yaml
<br>
the link of the sweep created along with the id will be provided after you run the above command, just add the argument --count after the command provided. following is an example of such a command
<br>

## wandb agent da24s002-indian-institute-of-technology-madras/DA6401_Assignment_3/q5ftm0js --count 35
<br>

run the above to start a wandb sweep.
by default the sweep uses the list of hyperparams written in config.yaml


========================================================================================
```yaml
program: "train_and_visualize.py"
name: "DA6401_Assignment_3"
method: "bayes"
metric:
  goal: maximize
  name: validation_accuracy
parameters:
  epochs:
    values: [10,15]
  model_type:
    values: ["basic", "attention"]
  encoder_rnn_type:
    values: ["LSTM","GRU"]
  decoder_rnn_type:
    values: ["LSTM","GRU"]
  embedding_dim:
    values: [64,256]
  hidden_dim:
    values: [64,256]
  encoder_layers:
    values: [2,3]
  decoder_layers:
    values: [2,3]
  batch_size:
    values: [16, 32, 64]
  dropout:
    values: [0,0.2,0.3]
  learning_rate: 
    max: 0.001
    min: 0.0001
    distribution: log_uniform_values
  clip:
    values: [1.0, 2.0, 5.0]
  teacher_forcing_ratio:
    values: [0.5, 1.0]
  # decode_method:
  #   values: ["beam", "greedy"]
  # beam_width:
  #   values: [3,5]

```
## Setting up the data  <br>
Please make sure you download the Dakshina dataset from the link: https://github.com/google-research-datasets/dakshina <br>
Extract the zip, and paste the folder: nature_12K where your train.py is:<br>
Your folder structure should look like this: <br>

_________________<br>
|&nbsp;&nbsp;train_visualize.py<br>
|&nbsp;&nbsp;config.yaml<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;dakshina_dataset_v1.0<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;dakshina_dataset_v1.0<br>
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;|&nbsp;&nbsp;hi<br>
|.<br>
|.<br>
|.<br>
_________________ <br>
The train_visualize.py file is the file containing the main method, and is the entry point to the training where we can pass our hyper-params<br>
via command line arguments. <br>
config.yaml is the hyper-parameter sweeping file, containing all options explored for hyper-param tuning. <br>
Seq2SeqRNN.py and Seq2SeqAttention.py contains the model definition which is being read by train_and_visualize.py to train the network. <br>
best_attention_hi_model.pt is the best model I have found using hyper-param tuning, this can be loaded with model.py<br>
and inferences can be made on any data without training.<br>
All the code for visualization of heatmap and other visualizations like confusion matrices can be found in the VisualizationUtils.py<br>
================== <br>

<br>
Link to Github Repository:<br>
https://github.com/da24s002/DA6401_Assignment_3_Submission<br>

========================================================================================<br>

Wandb report link: https://wandb.ai/da24s002-indian-institute-of-technology-madras/DA6401_Assignment_3/reports/DA6401-Assignment-3--VmlldzoxMjc3MTUwNQ
