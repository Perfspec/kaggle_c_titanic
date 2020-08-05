# kaggle_c_titanic
The aim of this program is to compete in the [Kaggle Titanic Survival Prediction Competition](https://www.kaggle.com/c/titanic/overview)

This programme will be a command line interface to generate prediction files (output.csv) based on a model that is trained on the embedded data/training.csv file and then applied to the embedded data/test.csv file.

Run this programme: `cargo run`

The contents of train.csv should be printed to the console, where the records have been loaded using the csv crate and deserialized using the serde crate.
This means that the records in train.csv must align with the struct TrainingRecord, as defined in lib.rs, or the application will throw an error.