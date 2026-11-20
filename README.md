# Modelling insurance charges in R

Can the same insurance records answer two different questions: how large a charge might be, and whether it crosses a fixed threshold? This analysis uses age, smoking status, BMI and other recorded characteristics to explore both questions on the public insurance dataset.

| Question | Models in the script | How to read the evaluation |
| --- | --- | --- |
| How accurately can the models estimate log charges? | Multiple linear regression, support vector regression and a regression tree | The regression section holds out half the records. RMSE is on the log-charge scale, so it cannot be read as a dollar error. |
| Which records have charges above 15,000? | Logistic regression and a classification tree | The classification section uses a separate 70/30 split and produces confusion matrices and ROC plots. The threshold is a choice made in the script. |

## Follow the analysis

Start with the [R script](Final%20Project.R). It creates a binary BMI-range feature, fits the models and generates diagnostic plots. The regression and classification sections use different outcomes and train/test splits, so their scores answer different questions.

## Reproduce locally

Download the insurance CSV from the [dataset page](https://www.kaggle.com/datasets/mirichoi0218/insurance). In an interactive R session, install the packages listed in the script's opening `library()` calls, then source `Final Project.R`. When the file chooser opens, select the downloaded CSV. Inspect the evaluation output and plots in the R session.
