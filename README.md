# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I chosen the following hpyerparameters. 

hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch-size": CategoricalParameter([64, 128, 256]),
    "epochs":CategoricalParameter([2, 4, 6]),
    "test-batch-size":CategoricalParameter([15, 30, 73])
}

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

Debugger Overview: i imported sagemaker debuger module in notebook. imported "Rule", "DebuggerHookConfig","rule_configs".
i aslo defined the rules like "vanishing_gradient", "overfit" etc. we also need hook config, so created a hook config. 
I specified the debugger rules and hook config in the estimator. 
in the training script, i created the hook object (please note, require smdebug module) and registered the hook to model. also, added the hook for train and evaluation modes. 

Profiling Overview: I imported ProfilerConfig, FrameworkProfile along with other debugger modules. created  a profile config and profile rules like "loss_not_decreasing" , "LowGPUUtilization" etc. 
i specified the profiling rules and config in the estimator. 


### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
Observation 1 : VanishingGradient: Error
The gradient in a train became extremely small.vanishing gradient prevents the weight from changning its value. it might stop NN from further trianing. 

In the profiler report , i noticed  CPUBottleneck , IOBottleneck etc.. along with recomendations which are useful for model training planning and development. 

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
Overview: i tried to deploy the trained model (which was trained along with debugger and profiler rule) but its failed due to "module import" issues. i created a seperate training script to train with best hpo without debugger and profiler. after that, i was able to deploy the model to endpoint. 
for quering, i loaded raw image and transformed into required format. and then used the method "predict" to get the model output. from the model output, fetch the maximumn value from the Output.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
uploaded to Git. filename : EndPoint-Screenshot.jpg

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
