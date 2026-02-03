# Experiments on MLP Model

## Experiment 0: Simple Values
**Hyperparameters**
| Hyperparameter     | Value |
| ---                | ---   |
| `N_EPOCHS`         | 10    |
| `N_HIDDENS`        | 2     |
| `N_SNEAKY_NEURONS` | 128   |  

**Evaluation Metrics**
- $\text{correct}/\text{total} = 0.8190$ 

**Remarks**. None. This is the preliminary test using relatively
simple values.

## Experiment 1: 50 Epochs
**Hyperparameters**
| Hyperparameter     | Value |
| ---                | ---   |
| `N_EPOCHS`         | 50    |
| `N_HIDDENS`        | 2     |
| `N_SNEAKY_NEURONS` | 128   |  

**Evaluation Metrics**
- $\text{correct}/\text{total} = 0.8083$ 

**Remarks**. The loss function was on epoch 46, with a value of 0.0105.

<!-- 
    ( ============================ )
    ( EXPERIMENT FORMAT ---------- )
    ( ---------------------------- )
    ( Fill in the buckets -------- )
    ( the "[_]" ------------------ )
    ( ---------------------------- )
    ( Copy everything starting --- )
    ( the line below ------------- )
    ( ============================ ) 

    ## Experiment X: <Simple Description>
    **Hyperparameters**
    | Hyperparameter     | Value |
    | ---                | ---   |
    | `N_EPOCHS`         | [_]   |
    | `N_HIDDENS`        | [_]   |
    | `N_SNEAKY_NEURONS` | [_]   |  

    **Evaluation Metrics**
    - $\text{correct}/\text{total} = [_] $
    [_ Add more here as needed]

    **Remarks**. [_ Add explanation here for the motivation for the 
    changes to the hyperparameters. And its effect on the evaluation
    metrics. Explicitly mention what experiment number you're 
    comparing against.]
 -->