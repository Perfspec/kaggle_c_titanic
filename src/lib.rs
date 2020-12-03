use std::error::Error;
use csv::Reader;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;

pub struct Config {
    pub learning_rate: f64,
	pub tolerance: f64,
    pub training_data_filename: String,
    pub test_data_filename: String,
    pub output_filename: String,
}

impl Config {
    pub fn new(args: &[String]) -> Result<Config, &'static str> {
        if args.len() < 6 {
            return Err("not enough arguments");
        }
        let learning_rate_string = args[1].clone();
        let tolerance_string = args[2].clone();
        let training_data_filename = args[3].clone();
        let test_data_filename = args[4].clone();
        let output_filename = args[5].clone();
        
        match learning_rate_string.parse::<f64>() {
            Ok(learning_rate) => {
				match tolerance_string.parse::<f64>() {
					Ok(tolerance) => Ok(Config {learning_rate, tolerance, training_data_filename, test_data_filename, output_filename}),
					Err(_) => Err("unable to parse tolerance"),
				}
			},
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
	
	//let mut cost = passenger_weights.cost()
    //while cost.gt(tolerance) {
	//	match passenger_weights.gradient_descent_update(&passengers) {
	//		Ok(_) => {},
	//		Err(_) => {}
	//	}
	//}
    
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

impl TrainingPassenger {
	pub fn new(
		passenger_id: u64,
		survived: Survived,
		passenger_class: PassengerClass,
		name: String,
		sex: Sex,
		age: f64,
		siblings_spouses: u8,
		parents_children: u8,
		ticket_id: String,
		fare: f64,
		cabin_id: String,
		port_of_embarkation: PortOfEmbarkation
	) -> TrainingPassenger {
		TrainingPassenger {
			passenger_id,
			survived,
			passenger_class: Some(passenger_class),
			name: Some(name),
			sex: Some(sex),
			age: Some(age),
			siblings_spouses: Some(siblings_spouses),
			parents_children: Some(parents_children),
			ticket_id: Some(ticket_id),
			fare: Some(fare),
			cabin_id: Some(cabin_id),
			port_of_embarkation: Some(port_of_embarkation),
		}
	}
	
	pub fn get_passenger_id(&self) -> &u64 {
		self.passenger_id
	}
	
	pub fn get_survived(&self) -> &Survived {
		self.survived
	}
	
	pub fn get_passenger_class(&self) -> &Option<PassengerClass> {
		self.passenger_class
	}
	
	pub fn get_name(&self) -> &Option<String> {
		self.name
	}
	
	pub fn get_sex(&self) -> &Option<Sex> {
		self.sex
	}
	
	pub fn get_age(&self) -> &Option<f64> {
		self.age
	}
	
	pub fn get_siblings_spouses(&self) -> &Option<u8> {
		self.siblings_spouses
	}
	
	pub fn get_parents_children(&self) -> &Option<u8> {
		self.parents_children
	}
	
	pub fn get_ticket_id(&self) -> &Option<String> {
		self.ticket_id
	}
	
	pub fn get_fare(&self) -> &Option<f64> {
		self.fare
	}
	
	pub fn get_cabin_id(&self) -> &Option<String> {
		self.cabin_id
	}
	
	pub fn get_port_of_embarkation(&self) -> &Option<PortOfEmbarkation> {
		self.port_of_embarkation
	}
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
    pub fn new() -> PassengerWeights {
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
        
        //Optional enums can be fully instantiated now, because they have a maximum number of weights.
        //One for each value in the enum when Optional matches Some and one for when Optional matches None.
        let mut passenger_class = Vec::with_capacity(4);
		passenger_class.push(1_f64);
		passenger_class.push(1_f64);
		passenger_class.push(1_f64);
		passenger_class.push(1_f64);
        
        let mut sex = Vec::with_capacity(3);
		sex.push(1_f64);
		sex.push(1_f64);
		sex.push(1_f64);
                
        let mut port_of_embarkation = Vec::with_capacity(4);
		port_of_embarkation.push(1_f64);
		port_of_embarkation.push(1_f64);
		port_of_embarkation.push(1_f64);
		port_of_embarkation.push(1_f64);
                
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
	
	pub fn hypothesis(&self, training_passenger: &TrainingPassenger) -> Result<f64, &'static str> {
		let mut weighted_sum = 0_f64;
		
		match *training_passenger.get_passenger_class() {
			None => {
				match self.passenger_class.get(0) {
					None => {
						Err("PassengerWeights::hypothesis: passenger_class weight 0 was unreachable")
					},
					Some(weight) => {
						weighted_sum.add(weight);
					},
				}
			},
			Some(passenger_class) => {
				match passenger_class {
					PassengerClass::First => {
						match self.passenger_class.get(1) {
							None => {
								Err("PassengerWeights::hypothesis: passenger_class weight 1 was unreachable")
							},
							Some(weight) => {
								weighted_sum.add(weight);
							},
						}
					},
					PassengerClass::Second => {
						match self.passenger_class.get(2) {
							None => {
								Err("PassengerWeights::hypothesis: passenger_class weight 2 was unreachable")
							},
							Some(weight) => {
								weighted_sum.add(weight);
							},
						}
					},
					PassengerClass::Third => {
						match self.passenger_class.get(3) {
							None => {
								Err("PassengerWeights::hypothesis: passenger_class weight 3 was unreachable")
							},
							Some(weight) => {
								weighted_sum.add(weight);
							},
						}
					},
				}
			},
		}
		
		match *training_passenger.get_sex() {
			None => {
				match self.sex.get(0) {
					None => {
						Err("PassengerWeights::hypothesis: sex weight 0 was unreachable")
					},
					Some(weight) => {
						weighted_sum.add(weight);
					},
				}
			},
			Some(sex) => {
				match sex {
					Sex::Female => {
						match self.sex.get(1) {
							None => {
								Err("PassengerWeights::hypothesis: sex weight 1 was unreachable")
							},
							Some(weight) => {
								weighted_sum.add(weight);
							},
						}
					},
					Sex::Male => {
						match self.sex.get(2) {
							None => {
								Err("PassengerWeights::hypothesis: sex weight 2 was unreachable")
							},
							Some(weight) => {
								weighted_sum.add(weight);
							},
						}
					},
				}
			},
		}
	}
	
	pub fn cost(&self, training_passenger: &TrainingPassenger) -> Result<f64, &'static str> {
		let mut cost = 0_f64;
		
		match *training_passenger.get_survived() {
			Survived::Yes => {
				match self.hypothesis(training_passenger) {
					Ok(hypothesis) => {
						cost.add(hypothesis.ln());
					},
					Err(error) => Err(error),
				}
			},
			Survived::No => {
				match self.hypothesis(training_passenger) {
					Ok(hypothesis) => {
						cost.add((1 - hypothesis).ln());
					},
					Err(error) => Err(error),
				}
			},
		}
		Ok(-cost)
	}
	
	pub fn avg_cost(&self, training_passengers: &Vec<TrainingPassenger>) -> Result<f64, String> {
		let mut sum = 0_f64;
		let mut counter = 0_f64;
		
		for training_passenger in *training_passengers {
			match *self.cost(&training_passenger) {
				Ok(cost) => {
					sum.add(cost);
				},
				Err(e) => {
					let passenger_id = *(training_passenger.get_passenger_id());
					let message = format!("PassengerWeights::avg_cost was unable to calculate cost for passenger_id: {}. {}", passenger_id, e);
					Err(message)
				},
			}
			counter.add(1_f64);
		}
		
		if counter.eq(0_f64) {
			let message = "PassengerWeights::avg_cost counter is a denominator and was zero".to_string();
			Err(message)
		}
		
		let avg = sum.div(counter);
		Ok(avg)
	}
	
	pub fn gradient_descent_update(self, learning_rate: f64, training_passengers: Vec<TrainingPassenger>) -> Result<(), String> {
		
	}
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[should_panic(expected = "not enough arguments")]
    fn when_less_than_6_arguments_then_return_error() {
        let args = vec!["first".to_string(), "second".to_string()];
        Config::new(&args).unwrap();
    }
    
    #[test]
    fn when_at_least_6_arguments_then_create_config() {
        let args = vec!["first".to_string(), "2".to_string(), "3".to_string(), "fourth".to_string(), "fifth".to_string(), "sixth".to_string()];
        let conf = Config::new(&args).unwrap();
        let mut sum_strings = String::new();
		let mut sum_nums = 0_f64;
		
		sum_nums.add(&conf.learning_rate);
        sum_nums.add(&conf.tolerance);
		
        sum_strings.push_str(&conf.training_data_filename);
		sum_strings.push_str("-");
        sum_strings.push_str(&conf.test_data_filename);
        sum_strings.push_str("-");
        sum_strings.push_str(&conf.output_filename);
		
		assert_eq!(&sum_nums, 5_f64);
        assert_eq!(&sum_strings, "fourth-fifth-sixth");
    }
	
	//#[test]
	//fn when_given_new_weights_and_record_then_performing_gradient_descent_update_produces_correct_output_weights() {
    //    // Initialize weights
    //    let mut passenger_weights = PassengerWeights::new();
	//	let mut passengers = Vec::new();
	//	let passenger = TrainingPassenger {
	//		
	//	}
	//	passenger.push(passenger);
	//	passenger_weights.gradient_descent_update(&passengers).unwrap();
	//	
	//	assert_eq!(passenger_weights.get_bias(), 10.0);
	//}
}