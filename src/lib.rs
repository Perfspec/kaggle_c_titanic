use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};
use std::collections::HashMap;

#[macro_use]
extern crate approx;

mod classification;

#[cfg(test)]
mod tests;

pub struct Config {
    learning_rate: f64,
    tolerance: f64,
    training_data_filename: String,
    test_data_filename: String,
    output_filename: String,
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
    
    pub fn get_learning_rate(&self) -> &f64 {
        &self.learning_rate
    }
    
    pub fn get_tolerance(&self) -> &f64 {
        &self.tolerance
    }
    
    pub fn get_training_data_filename(&self) -> &String {
        &self.training_data_filename
    }
    
    pub fn get_test_data_filename(&self) -> &String {
        &self.test_data_filename
    }
    
    pub fn get_output_filename(&self) -> &String {
        &self.output_filename
    }
	
	pub fn get_training_passengers(&self) -> Result<Vec<TrainingPassenger>, String> {
		//Read training_data into vector of training_passengers, which will be reused many times.
		let mut training_passengers = Vec::new();
		match Reader::from_path(self.get_training_data_filename()) {
			Ok(mut training_data) => {
				for result in training_data.deserialize() {
					match result {
						Ok(record) => {
							let training_passenger: TrainingPassenger = record;
							training_passengers.push(training_passenger);
						},
						Err(_) => return Err("Config::get_training_passengers Failed to deserialize TrainingPassenger".to_string()),
					}
				}
				println!("Config::get_training_passengers training_passengers: Vec<TrainingPassenger> has been instantiated with length {}", training_passengers.len());
				Ok(training_passengers)
			},
			Err(_) => {
				let message = format!("Config::get_training_passengers Failed to read from {}", &self.get_training_data_filename());
				Err(message)
			},
		}
	}
	
	pub fn get_test_passengers(&self) -> Result<Vec<Passenger>, String> {
		//Read test_data into vector of passengers, which will be tested once each.
		let mut test_passengers = Vec::new();
		match Reader::from_path(self.get_test_data_filename()) {
			Ok(mut test_data) => {
				for result in test_data.deserialize() {
					match result {
						Ok(record) => {
							let test_passenger: Passenger = record;
							test_passengers.push(test_passenger);
						},
						Err(_) => return Err("Config::get_test_passengers Failed to deserialize TestPassenger".to_string()),
					}
				}
				println!("Config::get_test_passengers test_passengers: Vec<TestPassenger> has been instantiated with length {}", test_passengers.len());
				Ok(test_passengers)
			},
			Err(_) => {
				let message = format!("Config::get_test_passengers Failed to read from {}", &self.get_test_data_filename());
				Err(message)
			},
		}
	}
	
	pub fn write_output(&self, passenger_weights: &PassengerWeights, test_passengers: &Vec<Passenger>) -> Result<(), String> {
		match Writer::from_path(self.get_output_filename()) {
			Ok(mut writer) => {
				for test_passenger in test_passengers {
					let outcome = classification::predict(passenger_weights, test_passenger)?;
					let tested_passenger = TestedPassenger::new(outcome);
					
					if let Err(e2) = writer.serialize(tested_passenger) {
						let message = format!("Config::write_output Failed to serialize TestedPassenger {}. Serde: {}", test_passenger.get_passenger_id(), e2);
						return Err(message)
					}
				}
				println!("Config::write_output Completed writing test results to {}", &self.get_output_filename());
				Ok(())
			},
			Err(e1) => {
				let message = format!("Config::write_output Failed to create writer to {}. Serde: {}", &self.get_output_filename(), e1);
				Err(message)
			},
		}
	}
}

pub fn run(config: &mut Config) -> Result<(), String> {
    
	let training_passengers = config.get_training_passengers()?;
	let test_passengers = config.get_test_passengers()?;
	
	// Initialize weights
	let mut passenger_weights = PassengerWeights::new();
	
	classification::solve(&training_passengers, &mut passenger_weights, *config.get_learning_rate(), config.get_tolerance())?;
	
	config.write_output(&passenger_weights, &test_passengers)
}

#[derive(Debug, Deserialize, Serialize)]
pub enum Survived {
    #[serde(rename = "0")]
    Yes,
    
    #[serde(rename = "1")]
    No
}

#[derive(Debug, Deserialize)]
pub enum PassengerClass {
    #[serde(rename = "1")]
    First,
    
    #[serde(rename = "2")]
    Second,
    
    #[serde(rename = "3")]
    Third,
}

#[derive(Debug, Deserialize)]
pub enum Sex {
    #[serde(rename = "male")]
    Male,
    
    #[serde(rename = "female")]
    Female
}

#[derive(Debug, Deserialize)]
pub enum PortOfEmbarkation {
    #[serde(rename = "C")]
    Cherbourg,
    
    #[serde(rename = "S")]
    Southampton,
    
    #[serde(rename = "Q")]
    Queenstown,
}

#[derive(Debug, Deserialize)]
pub struct TrainingPassenger {
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
    siblings_spouses: Option<usize>,
    
    #[serde(rename = "Parch")]
    parents_children: Option<usize>,
    
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
        siblings_spouses: usize,
        parents_children: usize,
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
        &self.passenger_id
    }
    
    pub fn get_survived(&self) -> &Survived {
        &self.survived
    }
    
    pub fn get_passenger_class(&self) -> &Option<PassengerClass> {
        &self.passenger_class
    }
    
    pub fn get_name(&self) -> &Option<String> {
        &self.name
    }
    
    pub fn get_sex(&self) -> &Option<Sex> {
        &self.sex
    }
    
    pub fn get_age(&self) -> &Option<f64> {
        &self.age
    }
    
    pub fn get_siblings_spouses(&self) -> &Option<usize> {
        &self.siblings_spouses
    }
    
    pub fn get_parents_children(&self) -> &Option<usize> {
        &self.parents_children
    }
    
    pub fn get_ticket_id(&self) -> &Option<String> {
        &self.ticket_id
    }
    
    pub fn get_fare(&self) -> &Option<f64> {
        &self.fare
    }
    
    pub fn get_cabin_id(&self) -> &Option<String> {
        &self.cabin_id
    }
    
    pub fn get_port_of_embarkation(&self) -> &Option<PortOfEmbarkation> {
        &self.port_of_embarkation
    }
}

#[derive(Debug, Deserialize)]
pub struct Passenger {
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
    siblings_spouses: Option<usize>,
    
    #[serde(rename = "Parch")]
    parents_children: Option<usize>,
    
    #[serde(rename = "Ticket")]
    ticket_id: Option<String>,
    
    #[serde(rename = "Fare")]
    fare: Option<f64>,
    
    #[serde(rename = "Cabin")]
    cabin_id: Option<String>,
    
    #[serde(rename = "Embarked")]
    port_of_embarkation: Option<PortOfEmbarkation>,
}

impl Passenger {
	pub fn new(
        passenger_id: u64,
        passenger_class: PassengerClass,
        name: String,
        sex: Sex,
        age: f64,
        siblings_spouses: usize,
        parents_children: usize,
        ticket_id: String,
        fare: f64,
        cabin_id: String,
        port_of_embarkation: PortOfEmbarkation
    ) -> Passenger {
        Passenger {
            passenger_id,
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
        &self.passenger_id
    }
    
    pub fn get_passenger_class(&self) -> &Option<PassengerClass> {
        &self.passenger_class
    }
    
    pub fn get_name(&self) -> &Option<String> {
        &self.name
    }
    
    pub fn get_sex(&self) -> &Option<Sex> {
        &self.sex
    }
    
    pub fn get_age(&self) -> &Option<f64> {
        &self.age
    }
    
    pub fn get_siblings_spouses(&self) -> &Option<usize> {
        &self.siblings_spouses
    }
    
    pub fn get_parents_children(&self) -> &Option<usize> {
        &self.parents_children
    }
    
    pub fn get_ticket_id(&self) -> &Option<String> {
        &self.ticket_id
    }
    
    pub fn get_fare(&self) -> &Option<f64> {
        &self.fare
    }
    
    pub fn get_cabin_id(&self) -> &Option<String> {
        &self.cabin_id
    }
    
    pub fn get_port_of_embarkation(&self) -> &Option<PortOfEmbarkation> {
        &self.port_of_embarkation
    }
}

#[derive(Debug, Serialize)]
pub struct TestedPassenger {
    #[serde(rename = "PassengerId")]
    passenger_id: u64,
    
    #[serde(rename = "Survived")]
    survived: Survived,
}

impl TestedPassenger {
	pub fn new(outcome: classification::Outcome) -> TestedPassenger {
		match outcome.prediction {
			classification::BinaryClass::Yes => {
				TestedPassenger {
					passenger_id: outcome.record_id,
					survived: Survived::Yes
				}
			},
			classification::BinaryClass::No => {
				TestedPassenger {
					passenger_id: outcome.record_id,
					survived: Survived::No
				}
			}
		}
		
	}
}

#[derive(Debug)]
pub struct PassengerWeights {
    bias: f64,
    passenger_class: HashMap<usize, f64>,
    name: HashMap<usize, f64>,
    sex: HashMap<usize, f64>,
    age: HashMap<usize, f64>,
    siblings_spouses: HashMap<usize, f64>,
    parents_children: HashMap<usize, f64>,
    ticket_id: HashMap<usize, f64>,
    fare: HashMap<usize, f64>,
    cabin_id: HashMap<usize, f64>,
    port_of_embarkation: HashMap<usize, f64>,
}

impl PassengerWeights {
    pub fn new() -> PassengerWeights {
        let bias = 1_f64;
        
        //Optional integers, floats, enums and string can be fully instantiated now, because they have a maximum number of weights.
        //At least one for when Optional matches Some and one for when Optional matches None.
        //Index 0 is always reserved for 
        let mut age = HashMap::new();
        age.insert(0, 1_f64);
        age.insert(1, 1_f64);
        
        let mut siblings_spouses = HashMap::new();
        siblings_spouses.insert(0, 1_f64);
        siblings_spouses.insert(1, 1_f64);
        
        let mut parents_children = HashMap::new();
        parents_children.insert(0, 1_f64);
        parents_children.insert(1, 1_f64);
        
        let mut fare = HashMap::new();
        fare.insert(0, 1_f64);
        fare.insert(1, 1_f64);
        
        //In future versions, the strings may be categorized, so capacity may be set to a fixed number of categories. TBD
        let mut name = HashMap::new();
        name.insert(0, 1_f64);
        name.insert(1, 1_f64);
        
        let mut ticket_id = HashMap::new();
        ticket_id.insert(0, 1_f64);
        ticket_id.insert(1, 1_f64);
        
        let mut cabin_id = HashMap::new();
        cabin_id.insert(0, 1_f64);
        cabin_id.insert(1, 1_f64);
        
        // Enums can have more than two categories in the Some option
        let mut passenger_class = HashMap::new();
        passenger_class.insert(0, 1_f64);
        passenger_class.insert(1, 1_f64);
        passenger_class.insert(2, 1_f64);
        passenger_class.insert(3, 1_f64);
        
        let mut sex = HashMap::new();
        sex.insert(0, 1_f64);
        sex.insert(1, 1_f64);
        sex.insert(2, 1_f64);
                
        let mut port_of_embarkation = HashMap::new();
        port_of_embarkation.insert(0, 1_f64);
        port_of_embarkation.insert(1, 1_f64);
        port_of_embarkation.insert(2, 1_f64);
        port_of_embarkation.insert(3, 1_f64);
                
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

impl classification::LogisticBinaryClassificationTestable for TrainingPassenger {
	type Weights = PassengerWeights;
	
    fn hypothesis(self: &Self, weights: &Self::Weights) -> Result<f64, String> {
        let mut weighted_sum = 0_f64;
        
        weighted_sum = weighted_sum.add(weights.bias);
        
        match self.get_name() {
            None => {
                match weights.name.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_name) => {
                match weights.name.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match self.get_age() {
            None => {
                match weights.age.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(age) => {
                match weights.age.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: age weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(age));
                    },
                }
            },
        }
        
        match self.get_siblings_spouses() {
            None => {
                match weights.siblings_spouses.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(siblings_spouses) => {
                match weights.siblings_spouses.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: siblings_spouses weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(classification::quick_convert(siblings_spouses)));
                    },
                }
            },
        }
        
        match self.get_parents_children() {
            None => {
                match weights.parents_children.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(parents_children) => {
                match weights.parents_children.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: parents_children weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(classification::quick_convert(parents_children)));
                    },
                }
            },
        }
        
        match self.get_fare() {
            None => {
                match weights.fare.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(fare) => {
                match weights.fare.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: fare weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(fare));
                    },
                }
            },
        }
        
        match self.get_ticket_id() {
            None => {
                match weights.ticket_id.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_ticket_id) => {
                match weights.ticket_id.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match self.get_cabin_id() {
            None => {
                match weights.cabin_id.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_cabin_id) => {
                match weights.cabin_id.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match self.get_passenger_class() {
            None => {
                match weights.passenger_class.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(passenger_class) => {
                match passenger_class {
                    PassengerClass::First => {
                        match weights.passenger_class.get(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match weights.passenger_class.get(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match weights.passenger_class.get(&3) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }
        
        match self.get_sex() {
            None => {
                match weights.sex.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: sex weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(sex) => {
                match sex {
                    Sex::Female => {
                        match weights.sex.get(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    Sex::Male => {
                        match weights.sex.get(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: sex weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }
        
        match self.get_port_of_embarkation() {
            None => {
                match weights.port_of_embarkation.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(port_of_embarkation) => {
                match port_of_embarkation {
                    PortOfEmbarkation::Cherbourg => {
                        match weights.port_of_embarkation.get(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match weights.port_of_embarkation.get(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match weights.port_of_embarkation.get(&3) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }

        Ok(Self::logistic(weighted_sum))
    }
	
	fn get_record_id(self: &Self) -> &u64 {
		self.get_passenger_id()
	}
}

impl classification::LogisticBinaryClassificationTestable for Passenger {
    type Weights = PassengerWeights;
	
    fn hypothesis(self: &Self, weights: &Self::Weights) -> Result<f64, String> {
        let mut weighted_sum = 0_f64;
        
        weighted_sum = weighted_sum.add(weights.bias);
        
        match self.get_name() {
            None => {
                match weights.name.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_name) => {
                match weights.name.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match self.get_age() {
            None => {
                match weights.age.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(age) => {
                match weights.age.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: age weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(age));
                    },
                }
            },
        }
        
        match self.get_siblings_spouses() {
            None => {
                match weights.siblings_spouses.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(siblings_spouses) => {
                match weights.siblings_spouses.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: siblings_spouses weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(classification::quick_convert(siblings_spouses)));
                    },
                }
            },
        }
        
        match self.get_parents_children() {
            None => {
                match weights.parents_children.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(parents_children) => {
                match weights.parents_children.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: parents_children weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(classification::quick_convert(parents_children)));
                    },
                }
            },
        }
        
        match self.get_fare() {
            None => {
                match weights.fare.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(fare) => {
                match weights.fare.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: fare weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(fare));
                    },
                }
            },
        }
        
        match self.get_ticket_id() {
            None => {
                match weights.ticket_id.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_ticket_id) => {
                match weights.ticket_id.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match self.get_cabin_id() {
            None => {
                match weights.cabin_id.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_cabin_id) => {
                match weights.cabin_id.get(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match self.get_passenger_class() {
            None => {
                match weights.passenger_class.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(passenger_class) => {
                match passenger_class {
                    PassengerClass::First => {
                        match weights.passenger_class.get(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match weights.passenger_class.get(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match weights.passenger_class.get(&3) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }
        
        match self.get_sex() {
            None => {
                match weights.sex.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: sex weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(sex) => {
                match sex {
                    Sex::Female => {
                        match weights.sex.get(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    Sex::Male => {
                        match weights.sex.get(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: sex weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }
        
        match self.get_port_of_embarkation() {
            None => {
                match weights.port_of_embarkation.get(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(port_of_embarkation) => {
                match port_of_embarkation {
                    PortOfEmbarkation::Cherbourg => {
                        match weights.port_of_embarkation.get(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match weights.port_of_embarkation.get(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match weights.port_of_embarkation.get(&3) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTestable::hypothesis: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }

        Ok(Self::logistic(weighted_sum))
    }
	
	fn get_record_id(self: &Self) -> &u64 {
		self.get_passenger_id()
	}
}

impl classification::LogisticBinaryClassificationTrainable for TrainingPassenger {
	fn answer(self: &Self) -> classification::BinaryClass {
		match self.get_survived() {
			Survived::Yes => classification::BinaryClass::Yes,
			Survived::No => classification::BinaryClass::No
		}
	}

	fn update_weights(self: &Self, diff: &f64, weights: &mut Self::Weights) -> Result<(), String>{
        weights.bias = weights.bias.add(diff);
        
        match self.get_name() {
            None => {
                match weights.name.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(_name) => {
                match weights.name.get_mut(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
        }
        
        match self.get_age() {
            None => {
                match weights.age.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(age) => {
                match weights.age.get_mut(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: age weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(age));
                    },
                }
            },
        }
        
        match self.get_siblings_spouses() {
            None => {
                match weights.siblings_spouses.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(siblings_spouses) => {
                match weights.siblings_spouses.get_mut(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: siblings_spouses weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(classification::quick_convert(siblings_spouses)));
                    },
                }
            },
        }
        
        match self.get_parents_children() {
            None => {
                match weights.parents_children.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(parents_children) => {
                match weights.parents_children.get_mut(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: parents_children weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(classification::quick_convert(parents_children)));
                    },
                }
            },
        }
        
        match self.get_fare() {
            None => {
                match weights.fare.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(fare) => {
                match weights.fare.get_mut(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: fare weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(fare));
                    },
                }
            },
        }
        
        match self.get_ticket_id() {
            None => {
                match weights.ticket_id.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(_ticket_id) => {
                match weights.ticket_id.get_mut(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
        }
        
        match self.get_cabin_id() {
            None => {
                match weights.cabin_id.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(_cabin_id) => {
                match weights.cabin_id.get_mut(&1) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
        }
        
        match self.get_passenger_class() {
            None => {
                match weights.passenger_class.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(passenger_class) => {
                match passenger_class {
                    PassengerClass::First => {
                        match weights.passenger_class.get_mut(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match weights.passenger_class.get_mut(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match weights.passenger_class.get_mut(&3) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                }
            },
        }
        
        match self.get_sex() {
            None => {
                match weights.sex.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: sex weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(sex) => {
                match sex {
                    Sex::Female => {
                        match weights.sex.get_mut(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    Sex::Male => {
                        match weights.sex.get_mut(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: sex weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                }
            },
        }
        
        match self.get_port_of_embarkation() {
            None => {
                match weights.port_of_embarkation.get_mut(&0) {
                    None => {
                        let passenger_id = self.get_passenger_id();
                        let message = format!("LogisticBinaryClassificationTrainable::update_weights: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(port_of_embarkation) => {
                match port_of_embarkation {
                    PortOfEmbarkation::Cherbourg => {
                        match weights.port_of_embarkation.get_mut(&1) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match weights.port_of_embarkation.get_mut(&2) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match weights.port_of_embarkation.get_mut(&3) {
                            None => {
                                let passenger_id = self.get_passenger_id();
                                let message = format!("LogisticBinaryClassificationTrainable::update_weights: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                }
            },
        }

        Ok(())
    }
}

impl Clone for PassengerWeights {
    fn clone(&self) -> Self {
        let bias = self.bias.clone();
        let passenger_class = self.passenger_class.clone();
        let name = self.name.clone();
        let sex = self.sex.clone();
        let age = self.age.clone();
        let siblings_spouses = self.siblings_spouses.clone();
        let parents_children = self.parents_children.clone();
        let ticket_id = self.ticket_id.clone();
        let fare = self.fare.clone();
        let cabin_id = self.cabin_id.clone();
        let port_of_embarkation = self.port_of_embarkation.clone();
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