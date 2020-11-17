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
	
	// Initialize pmfs
	let mut passenger_class_pmf = HashMap::new();
	let mut sex_pmf = HashMap::new();
	let mut age_pmf = HashMap::new();
	let mut siblings_spouses_pmf = HashMap::new();
	
	for result in training_data.deserialize() {
        let record: TrainingRecord = result?;
        record.update_passenger_class_pmf(&mut passenger_class_pmf);
		record.update_sex_pmf(&mut sex_pmf);
		record.update_age_pmf(&mut age_pmf);
		record.update_siblings_spouses_pmf(&mut siblings_spouses_pmf);
    }
	
	println!("passenger_class_pmf: {:?}", passenger_class_pmf);
	println!("sex_pmf: {:?}", sex_pmf);
	println!("age_pmf: {:?}", age_pmf);
	println!("siblings_spouses_pmf: {:?}", siblings_spouses_pmf);
	
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
	fn update_passenger_class_pmf(&self, pmf: &mut HashMap<String, u64>) {
		match self.passenger_class {
			None => (),
			Some(PassengerClass::First) => {
				update_string_pmf(&"first".to_string(), pmf);
			},
			Some(PassengerClass::Second) => {
				update_string_pmf(&"second".to_string(), pmf);
			},
			Some(PassengerClass::Third) => {
				update_string_pmf(&"third".to_string(), pmf);
			},
		}
	}
	
	fn update_sex_pmf(&self, pmf: &mut HashMap<String, u64>) {
		match self.sex {
			None => (),
			Some(Sex::Male) => {
				update_string_pmf(&"male".to_string(), pmf);
			},
			Some(Sex::Female) => {
				update_string_pmf(&"female".to_string(), pmf);
			},
		}
	}
	
	fn update_age_pmf(&self, pmf: &mut HashMap<u64, u64>) {
		match self.age {
			None => (),
			Some(age) => {
				let rounded_age = age.round() as u64;
				update_u64_pmf(&rounded_age, pmf);
			}
		}
	}
	
	fn update_siblings_spouses_pmf(&self, pmf: &mut HashMap<u8, u64>) {
		match self.siblings_spouses {
			None => (),
			Some(siblings_spouses) => {
				update_u8_pmf(&siblings_spouses, pmf);
			}
		}
	}
	
	fn update_parents_children_pmf(&self, pmf: &mut HashMap<u8, u64>) {
		match self.parents_children {
			None => (),
			Some(parents_children) => {
				update_u8_pmf(&parents_children, pmf);
			}
		}
	}
	
	fn update_fare_pmf(&self, pmf: &mut HashMap<u64, u64>) {
		match self.fare {
			None => (),
			Some(fare) => {
				let rounded_fare = fare.round() as u64;
				update_u64_pmf(&rounded_fare, pmf);
			}
		}
	}
	
	
}

fn update_string_pmf(key: &String, pmf: &mut HashMap<String, u64>) {
	let count = pmf.entry(key.clone()).or_insert(0);
	*count += 1;
}

fn update_u64_pmf(key: &u64, pmf: &mut HashMap<u64, u64>) {
	let count = pmf.entry(*key).or_insert(0);
	*count += 1;
}

fn update_u8_pmf(key: &u8, pmf: &mut HashMap<u8, u64>) {
	let count = pmf.entry(*key).or_insert(0);
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