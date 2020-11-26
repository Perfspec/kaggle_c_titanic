use std::error::Error;
use csv::Reader;
use serde::Deserialize;
use std::collections::HashMap;

pub struct Config {
    pub learning_rate: f64,
    pub training_data_filename: String,
    pub test_data_filename: String,
    pub output_filename: String,
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 5 {
            return Err("not enough arguments");
        }
        let learning_rate_string = args[1].clone();
        let training_data_filename = args[2].clone();
        let test_data_filename = args[3].clone();
        let output_filename = args[4].clone();
        
        match learning_rate_string.parse::<f64>().unwrap() {
            Ok(learning_rate) => Ok(Config {learning_rate, training_data_filename, test_data_filename, output_filename}),
            Err(_) => Err("unable to parse learning rate"),
        }
    }
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    //Read training_data into vector of training_passengers, which will be reused many times.
    let mut training_data = Reader::from_path(config.training_data_filename)?;
    let mut training_passengers = Vec::new();
    
    for result in training_data.deserialize() {
        let mut training_passenger: TrainingPassenger = result?;
        training_passengers.push(training_passenger);
    }
    
    // Initialize weights
    let mut passenger_weights = PassengerWeights::new();
    
    
    Ok(())
}

#[derive(Debug, Deserialize)]
enum Survived {
    #[serde(rename = "0")]
    Yes,
    
    #[serde(rename = "1")]
    No
}

#[derive(Debug, Deserialize)]
enum PassengerClass {
    #[serde(rename = "1")]
    First,
    
    #[serde(rename = "2")]
    Second,
    
    #[serde(rename = "3")]
    Third,
}

#[derive(Debug, Deserialize)]
enum Sex {
    #[serde(rename = "male")]
    Male,
    
    #[serde(rename = "female")]
    Female
}

#[derive(Debug, Deserialize)]
enum PortOfEmbarkation {
    #[serde(rename = "C")]
    Cherbourg,
    
    #[serde(rename = "S")]
    Southampton,
    
    #[serde(rename = "Q")]
    Queenstown,
}

#[derive(Debug, Deserialize)]
struct TrainingPassenger {
    #[serde(rename = "PassengerId")]
    passenger_id: u64,
    
    #[serde(rename = "Survived")]
    survived: Survived,
    
    #[serde(rename = "Pclass")]
    passenger_class: Option<PassengerClass>,
    
    #[serde(rename = "Name")]
    name: Option<String>,
    
    #[serde(rename = "Sex")]
    sex: Option<Sex>,
    
    #[serde(rename = "Age")]
    age: Option<f64>,
    
    #[serde(rename = "SibSp")]
    siblings_spouses: Option<u8>,
    
    #[serde(rename = "Parch")]
    parents_children: Option<u8>,
    
    #[serde(rename = "Ticket")]
    ticket_id: Option<String>,
    
    #[serde(rename = "Fare")]
    fare: Option<f64>,
    
    #[serde(rename = "Cabin")]
    cabin_id: Option<String>,
    
    #[serde(rename = "Embarked")]
    port_of_embarkation: Option<PortOfEmbarkation>,
}

#[derive(Debug, Deserialize)]
struct Passenger {
    #[serde(rename = "PassengerId")]
    passenger_id: u64,
    
    #[serde(rename = "Pclass")]
    passenger_class: Option<PassengerClass>,
    
    #[serde(rename = "Name")]
    name: Option<String>,
    
    #[serde(rename = "Sex")]
    sex: Option<Sex>,
    
    #[serde(rename = "Age")]
    age: Option<f64>,
    
    #[serde(rename = "SibSp")]
    siblings_spouses: Option<u8>,
    
    #[serde(rename = "Parch")]
    parents_children: Option<u8>,
    
    #[serde(rename = "Ticket")]
    ticket_id: Option<String>,
    
    #[serde(rename = "Fare")]
    fare: Option<f64>,
    
    #[serde(rename = "Cabin")]
    cabin_id: Option<String>,
    
    #[serde(rename = "Embarked")]
    port_of_embarkation: Option<PortOfEmbarkation>,
}

#[derive(Debug)]
struct PassengerWeights {
    bias: f64,
    passenger_class: Vec<f64>,
    name: Vec<f64>,
    sex: Vec<f64>,
    age: Vec<f64>,
    siblings_spouses: Vec<f64>,
    parents_children: Vec<f64>,
    ticket_id: Vec<f64>,
    fare: Vec<f64>,
    cabin_id: Vec<f64>,
    port_of_embarkation: Vec<f64>,
}

impl PassengerWeights {
    fn new() -> PassengerWeights {
        let mut bias = 1_f64;
        
        //Optional floats, integers and strings only have one param instantiated for when Optional matches None.
        //When Optional matches Some(value), then extra capacity will be reserved at runtime.
        let mut name = Vec::new();
        name.push(1_f64);
        
        let mut age = Vec::new();
        age.push(1_f64);
        
        let mut siblings_spouses = Vec::new();
        siblings_spouses.push(1_f64);
        
        let mut parents_children = Vec::new();
        parents_children.push(1_f64);
        
        let mut ticket_id = Vec::new();
        ticket_id.push(1_f64);
        
        let mut cabin_id = Vec::new();
        cabin_id.push(1_f64);
        
        //Optional enums can be instantiated now, because they have a maximum number of weights.
        //One for each value in the enum when Optional matches Some and one for when Optional matches None.
        let mut passenger_class = Vec::with_capacity(4);
        for i in passenger_class {
            i=1_f64;
        }
        
        let mut sex = Vec::with_capacity(3);
        for i in sex {
            i=1_f64;
        }
                
        let mut port_of_embarkation = Vec::with_capacity(4);
        for i in port_of_embarkation {
            i=1_f64;
        }
                
        PassengerWeights {
            bias,
            passenger_class,
            name,
            sex,
            age,
            siblings_spouses,
            parents_children,
            ticket_id,
            fare,
            cabin_id,
            port_of_embarkation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[should_panic(expected = "not enough arguments")]
    fn when_less_than_four_arguments_then_return_error() {
        let args = vec!["first".to_string(), "second".to_string()];
        Config::new(&args).unwrap();
    }
    
    #[test]
    fn when_at_least_three_arguments_then_create_config() {
        let args = vec!["first".to_string(), "second".to_string(), "third".to_string(), "fourth".to_string()];
        let conf = Config::new(&args).unwrap();
        let mut sum = String::new();
        sum.push_str(&conf.training_data_filename);
        sum.push_str(&conf.test_data_filename);
        sum.push_str(&conf.output_filename);
        assert_eq!(&sum, "secondthirdfourth");
    }
}