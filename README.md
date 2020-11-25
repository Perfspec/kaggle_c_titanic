# kaggle_c_titanic
The aim of this program is to compete in the [Kaggle Titanic Survival Prediction Competition](https://www.kaggle.com/c/titanic/overview)

This programme will be a command line interface to generate prediction files (output.csv) based on a model that is trained on the embedded data/training.csv file and then applied to the embedded data/test.csv file.

Run this programme: `cargo run`

Upload docs/main.tex to [Some Latex Runtime](https://www.overleaf.com/) to see mathematical documentation for this programme.

The contents of train.csv should be printed to the console, where the records have been loaded using the csv crate and deserialized using the serde crate.
This means that the records in train.csv must align with the struct TrainingRecord, as defined in lib.rs, or the application will throw an error.

Recall conditional probability notation: P(A|B) is the probability of A given B

We begin our analysis by counting the proportion of people that survived given various scenarios in the record.
i.e. P(Survived|Scenario) for various scenarios.

These fields are discrete, i.e. None or Some(x), where x is an enum or integer.
We'll make a hashmap of the with the values of the cumulative distribution function.
1. passenger_class
2. sex
3. number of cabins
4. siblings_spouses
5. parents_children

These fields are continuous, i.e. None or Some(x), where x is a float.
We'll make a hashmap of the with the values of the cumulative distribution function on a grid.
1. age
2. fare

Also, the cabin_id may have spatial data that shows that a certain part of the boat was damaged more seriously, or more dangerous.
1. cabin_floor. Lower floors probably more dangerous?
2. cabin_rooms on each floor. Some sections of a floor may have been damaged, whereas the other section of a floor not so much?
3. number of cabins. Some passengers seemed to pay for multiple cabins, this may have meant more exits and so safer?

This could be enriched by gathering records of the ship's layout and modelling the positions of the cabins.
