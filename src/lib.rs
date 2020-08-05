use std::error::Error;
use csv::Reader;
use serde::Deserialize;
use std::collections::HashMap;

pub struct Config {
    pub training_data_filename: String,
	pub test_data_filename: String,
	pub output_filename: String,
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 4 {
            return Err("not enough arguments");
        }
        let training_data_filename = args[1].clone();
		let test_data_filename = args[2].clone();
		let output_filename = args[3].clone();
		
		Ok(Config {training_data_filename, test_data_filename, output_filename})
    }
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let mut training_data = Reader::from_path(config.training_data_filename)?;
	
	// Initialize cdfs
	let mut passenger_class_cdf = HashMap::new();
	
	for result in training_data.deserialize() {
        let record: TrainingRecord = result?;
        record.update_passenger_class_cdf(&mut passenger_class_cdf);
    }
	
	println!("passenger_class_cdf: {:?}", passenger_class_cdf);
	
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
struct TrainingRecord {
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

impl TrainingRecord {
	fn update_passenger_class_cdf(&self, cdf: &mut HashMap<String, u64>) {
		match self.passenger_class {
			None => (),
			Some(PassengerClass::First) => {
				update_cdf(&"first".to_string(), cdf);
			},
			Some(PassengerClass::Second) => {
				update_cdf(&"second".to_string(), cdf);
			},
			Some(PassengerClass::Third) => {
				update_cdf(&"third".to_string(), cdf);
			},
		}
	}
}

fn update_cdf(key: &String, cdf: &mut HashMap<String, u64>) {
	let count = cdf.entry(key.clone()).or_insert(1);
	*count += 1;
}

#[derive(Debug, Deserialize)]
struct Record {
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