use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Div};
use std::collections::HashMap;

#[macro_use]
extern crate approx;

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
    
    pub fn set_learning_rate(&mut self, new_val: f64) {
        self.learning_rate = new_val;
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
					let test_result = passenger_weights.test_hypothesis(test_passenger)?;
					let tested_passenger = TestedPassenger::new(*test_passenger.get_passenger_id(), test_result);
					
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
	
	match passenger_weights.avg_cost(&training_passengers) {
		Ok(num) => {
			let mut avg_cost = num;
			let mut num_iterations = 0_u64;
			println!("At iteration {}, the avg_cost is {}", num_iterations, avg_cost);
			
			while avg_cost.gt(config.get_tolerance()) {
				match passenger_weights.gradient_descent_update(config.get_learning_rate(), &training_passengers) {
					Ok(_) => {
						num_iterations = num_iterations.add(1_u64);
						match passenger_weights.avg_cost(&training_passengers) {
							Ok(num) => {
								if avg_cost.lt(&num) {
									&mut config.set_learning_rate(config.get_learning_rate().div(100_f64));
									println!("Learning rate divided by 10 at iteration {}. New learning_rate: {}", &num_iterations, config.get_learning_rate());
								}
								avg_cost = num;
								println!("At iteration {}, the avg_cost is {}", &num_iterations, &avg_cost);
							},
							Err(error) => return Err(error),
						}
					},
					Err(error) => return Err(error),
				}
			}
			println!("Tolerable Cost Achieved: passenger_weights={:#?}", &passenger_weights);
		},
		Err(error) => return Err(error),
	}
	
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
	pub fn new(
        passenger_id: u64,
        survived: Survived
	) -> TestedPassenger {
		TestedPassenger {
			passenger_id,
			survived
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
    
    pub fn hypothesis(&self, training_passenger: &TrainingPassenger) -> Result<f64, String> {
        let mut weighted_sum = 0_f64;
        
        weighted_sum = weighted_sum.add(self.bias);
        
        match training_passenger.get_name() {
            None => {
                match self.name.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_name) => {
                match self.name.get(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match training_passenger.get_age() {
            None => {
                match self.age.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(age) => {
                match self.age.get(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: age weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(age));
                    },
                }
            },
        }
        
        match training_passenger.get_siblings_spouses() {
            None => {
                match self.siblings_spouses.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(siblings_spouses) => {
                match self.siblings_spouses.get(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: siblings_spouses weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(PassengerWeights::quick_convert(siblings_spouses)));
                    },
                }
            },
        }
        
        match training_passenger.get_parents_children() {
            None => {
                match self.parents_children.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(parents_children) => {
                match self.parents_children.get(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: parents_children weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(PassengerWeights::quick_convert(parents_children)));
                    },
                }
            },
        }
        
        match training_passenger.get_fare() {
            None => {
                match self.fare.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(fare) => {
                match self.fare.get(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: fare weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(fare));
                    },
                }
            },
        }
        
        match training_passenger.get_ticket_id() {
            None => {
                match self.ticket_id.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_ticket_id) => {
                match self.ticket_id.get(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match training_passenger.get_cabin_id() {
            None => {
                match self.cabin_id.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_cabin_id) => {
                match self.cabin_id.get(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match training_passenger.get_passenger_class() {
            None => {
                match self.passenger_class.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.passenger_class.get(&1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match self.passenger_class.get(&2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match self.passenger_class.get(&3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
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
        
        match training_passenger.get_sex() {
            None => {
                match self.sex.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: sex weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.sex.get(&1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    Sex::Male => {
                        match self.sex.get(&2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: sex weight 2 was unreachable for passenger {}", passenger_id);
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
        
        match training_passenger.get_port_of_embarkation() {
            None => {
                match self.port_of_embarkation.get(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.port_of_embarkation.get(&1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match self.port_of_embarkation.get(&2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match self.port_of_embarkation.get(&3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
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

        Ok(1_f64.div(weighted_sum.exp().add(1_f64)))
    }
    
    pub fn cost(&self, training_passenger: &TrainingPassenger) -> Result<f64, String> {
        match training_passenger.get_survived() {
            Survived::Yes => {
                match self.hypothesis(training_passenger) {
                    Ok(hypothesis) => Ok(-(hypothesis.ln())),
                    Err(error) => Err(error),
                }
            },
            Survived::No => {
                match self.hypothesis(training_passenger) {
                    Ok(hypothesis) => Ok(-((1_f64 - hypothesis).ln())),
                    Err(error) => Err(error)
                }
            },
        }
    }
    
    pub fn avg_cost(&self, training_passengers: &Vec<TrainingPassenger>) -> Result<f64, String> {
        let mut sum = 0_f64;
        let mut counter = 0_f64;
        
        for training_passenger in training_passengers {
            match self.cost(training_passenger) {
                Ok(cost) => {
                    sum = sum.add(cost);
                    counter = counter.add(1_f64);
                },
                Err(e) => {
                    let passenger_id = training_passenger.get_passenger_id();
                    let message = format!("PassengerWeights::avg_cost was unable to calculate cost for passenger_id: {}. {}", passenger_id, e);
                    return Err(message)
                },
            }
        }
        
        if counter.eq(&0_f64) {
            let message = "PassengerWeights::avg_cost counter is a denominator and was zero".to_string();
            return Err(message)
        }
        
        Ok(sum.div(counter))
    }
    
    pub fn gradient_descent_update(&mut self, learning_rate: &f64, training_passengers: &Vec<TrainingPassenger>) -> Result<(), String> {
        let passenger_weights = self.clone();
        for training_passenger in training_passengers {
            match passenger_weights.diff_hypothesis(&training_passenger) {
                Ok(diff) => {
                    self.add(&(diff.mul(-learning_rate)), &training_passenger)?;
                },
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    
    fn diff_hypothesis(&self, training_passenger: &TrainingPassenger) -> Result<f64, String> {
        match *training_passenger.get_survived() {
            Survived::Yes => {
                match self.hypothesis(training_passenger) {
                    Ok(hypothesis) => Ok(1_f64 - hypothesis),
                    Err(error) => Err(error),
                }
            },
            Survived::No => {
                self.hypothesis(training_passenger)
            },
        }
    }
    
    fn add(&mut self, diff: &f64, training_passenger: &TrainingPassenger) -> Result<(), String> {
        self.bias = self.bias.add(diff);
        
        match training_passenger.get_name() {
            None => {
                match self.name.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(_name) => {
                match self.name.get_mut(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
        }
        
        match training_passenger.get_age() {
            None => {
                match self.age.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(age) => {
                match self.age.get_mut(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: age weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(age));
                    },
                }
            },
        }
        
        match training_passenger.get_siblings_spouses() {
            None => {
                match self.siblings_spouses.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(siblings_spouses) => {
                match self.siblings_spouses.get_mut(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: siblings_spouses weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(PassengerWeights::quick_convert(siblings_spouses)));
                    },
                }
            },
        }
        
        match training_passenger.get_parents_children() {
            None => {
                match self.parents_children.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(parents_children) => {
                match self.parents_children.get_mut(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: parents_children weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(PassengerWeights::quick_convert(parents_children)));
                    },
                }
            },
        }
        
        match training_passenger.get_fare() {
            None => {
                match self.fare.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(fare) => {
                match self.fare.get_mut(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: fare weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff.mul(fare));
                    },
                }
            },
        }
        
        match training_passenger.get_ticket_id() {
            None => {
                match self.ticket_id.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(_ticket_id) => {
                match self.ticket_id.get_mut(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
        }
        
        match training_passenger.get_cabin_id() {
            None => {
                match self.cabin_id.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
            Some(_cabin_id) => {
                match self.cabin_id.get_mut(&1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        *weight = weight.add(diff);
                    },
                }
            },
        }
        
        match training_passenger.get_passenger_class() {
            None => {
                match self.passenger_class.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.passenger_class.get_mut(&1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match self.passenger_class.get_mut(&2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match self.passenger_class.get_mut(&3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
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
        
        match training_passenger.get_sex() {
            None => {
                match self.sex.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: sex weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.sex.get_mut(&1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    Sex::Male => {
                        match self.sex.get_mut(&2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: sex weight 2 was unreachable for passenger {}", passenger_id);
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
        
        match training_passenger.get_port_of_embarkation() {
            None => {
                match self.port_of_embarkation.get_mut(&0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.port_of_embarkation.get_mut(&1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match self.port_of_embarkation.get_mut(&2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                *weight = weight.add(diff);
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match self.port_of_embarkation.get_mut(&3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
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
    
    fn quick_convert(num: &usize) -> f64 {
        let mut result = 0_f64;
        for _number in 0..*num {
            result = result.add(1_f64);
        }
        result
    }
	
	pub fn test_hypothesis(&self, test_passenger: &Passenger) -> Result<Survived, String> {
        let mut weighted_sum = 0_f64;
        
        weighted_sum = weighted_sum.add(self.bias);
        
        match test_passenger.get_name() {
            None => {
                match self.name.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_name) => {
                match self.name.get(&1) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match test_passenger.get_age() {
            None => {
                match self.age.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(age) => {
                match self.age.get(&1) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: age weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(age));
                    },
                }
            },
        }
        
        match test_passenger.get_siblings_spouses() {
            None => {
                match self.siblings_spouses.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(siblings_spouses) => {
                match self.siblings_spouses.get(&1) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: siblings_spouses weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(PassengerWeights::quick_convert(siblings_spouses)));
                    },
                }
            },
        }
        
        match test_passenger.get_parents_children() {
            None => {
                match self.parents_children.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(parents_children) => {
                match self.parents_children.get(&1) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: parents_children weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(PassengerWeights::quick_convert(parents_children)));
                    },
                }
            },
        }
        
        match test_passenger.get_fare() {
            None => {
                match self.fare.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(fare) => {
                match self.fare.get(&1) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: fare weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight.mul(fare));
                    },
                }
            },
        }
        
        match test_passenger.get_ticket_id() {
            None => {
                match self.ticket_id.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_ticket_id) => {
                match self.ticket_id.get(&1) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match test_passenger.get_cabin_id() {
            None => {
                match self.cabin_id.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
            Some(_cabin_id) => {
                match self.cabin_id.get(&1) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum = weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match test_passenger.get_passenger_class() {
            None => {
                match self.passenger_class.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.passenger_class.get(&1) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match self.passenger_class.get(&2) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match self.passenger_class.get(&3) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
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
        
        match test_passenger.get_sex() {
            None => {
                match self.sex.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: sex weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.sex.get(&1) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    Sex::Male => {
                        match self.sex.get(&2) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: sex weight 2 was unreachable for passenger {}", passenger_id);
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
        
        match test_passenger.get_port_of_embarkation() {
            None => {
                match self.port_of_embarkation.get(&0) {
                    None => {
                        let passenger_id = test_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::test_hypothesis: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
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
                        match self.port_of_embarkation.get(&1) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match self.port_of_embarkation.get(&2) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum = weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match self.port_of_embarkation.get(&3) {
                            None => {
                                let passenger_id = test_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::test_hypothesis: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
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

        let real_hypothesis = 1_f64.div(weighted_sum.exp().add(1_f64));
		if real_hypothesis > 0.5 {
			Ok(Survived::Yes)
		} else {
			Ok(Survived::No)
		}
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
        
        sum_nums = sum_nums.add(conf.get_learning_rate());
        sum_nums = sum_nums.add(conf.get_tolerance());
        
        sum_strings.push_str(&conf.training_data_filename);
        sum_strings.push_str("-");
        sum_strings.push_str(&conf.test_data_filename);
        sum_strings.push_str("-");
        sum_strings.push_str(&conf.output_filename);
        
        assert_abs_diff_eq!(sum_nums, 5_f64);
        assert_eq!(&sum_strings, "fourth-fifth-sixth");
    }
    
    #[test]
    fn when_ref_usize_get_quick_convert_to_f64() {
        assert_abs_diff_eq!(PassengerWeights::quick_convert(&23_usize),23_f64);
    }
    
    #[test]
    fn when_new_passenger_weights_and_training_passenger_then_get_hypothesis() {
        // Initialize weights
        let passenger_weights = PassengerWeights::new();
        let training_passenger = TrainingPassenger::new(
            1_u64,
            Survived::Yes,
            PassengerClass::First,
            "Lewis Webb".to_string(),
            Sex::Male,
            25.33_f64,
            3_usize,
            2_usize,
            "Golden Ticket".to_string(),
            45.67_f64,
            "1".to_string(),
            PortOfEmbarkation::Southampton
        );
        assert_abs_diff_eq!(passenger_weights.hypothesis(&training_passenger).unwrap(), 1_f64.div(83_f64.exp().add(1_f64)));
    }
    
    #[test]
    fn when_new_passenger_weights_and_training_passenger_then_get_diff_hypothesis() {
        // Initialize weights
        let passenger_weights = PassengerWeights::new();
        let training_passenger = TrainingPassenger::new(
            1_u64,
            Survived::Yes,
            PassengerClass::First,
            "Lewis Webb".to_string(),
            Sex::Male,
            25.33_f64,
            3_usize,
            2_usize,
            "Golden Ticket".to_string(),
            45.67_f64,
            "1".to_string(),
            PortOfEmbarkation::Southampton
        );
        assert_abs_diff_eq!(passenger_weights.diff_hypothesis(&training_passenger).unwrap(), 1_f64 - 1_f64.div(83_f64.exp().add(1_f64)));
    }
    
    #[test]
    fn when_new_passenger_weights_and_training_passenger_then_get_cost() {
        // Initialize weights
        let passenger_weights = PassengerWeights::new();
        let training_passenger = TrainingPassenger::new(
            1_u64,
            Survived::Yes,
            PassengerClass::First,
            "Lewis Webb".to_string(),
            Sex::Male,
            25.33_f64,
            3_usize,
            2_usize,
            "Golden Ticket".to_string(),
            45.67_f64,
            "1".to_string(),
            PortOfEmbarkation::Southampton
        );
        assert_abs_diff_eq!(passenger_weights.cost(&training_passenger).unwrap(), -(1_f64.div(83_f64.exp().add(1_f64))).ln());
    }
    
    #[test]
    fn when_new_passenger_weights_and_training_passenger_then_get_avg_cost() {
        // Initialize weights
        let passenger_weights = PassengerWeights::new();
        let training_passenger = TrainingPassenger::new(
            1_u64,
            Survived::Yes,
            PassengerClass::First,
            "Lewis Webb".to_string(),
            Sex::Male,
            25.33_f64,
            3_usize,
            2_usize,
            "Golden Ticket".to_string(),
            45.67_f64,
            "1".to_string(),
            PortOfEmbarkation::Southampton
        );
        let mut training_passengers = Vec::new();
        training_passengers.push(training_passenger);
        assert_abs_diff_eq!(passenger_weights.avg_cost(&training_passengers).unwrap(), -(1_f64.div(83_f64.exp().add(1_f64))).ln());
    }
    
    #[test]
    fn when_new_passenger_weights_and_training_passenger_and_add_1_to_all_weights_then_get_hypothesis() {
        // Initialize weights
        let mut passenger_weights = PassengerWeights::new();
        let training_passenger = TrainingPassenger::new(
            1_u64,
            Survived::Yes,
            PassengerClass::First,
            "Lewis Webb".to_string(),
            Sex::Male,
            25.33_f64,
            3_usize,
            2_usize,
            "Golden Ticket".to_string(),
            45.67_f64,
            "1".to_string(),
            PortOfEmbarkation::Southampton
        );
        let mut training_passengers = Vec::new();
        training_passengers.push(training_passenger);
        passenger_weights.gradient_descent_update(&(-1_f64), &training_passengers).unwrap();
        match training_passengers.get(0) {
            None => panic!("tests::when_new_passenger_weights_and_training_passenger_and_gradient_descent_update_then_get_hypothesis could not find item in vec"),
            Some(training_passenger0) => assert_abs_diff_eq!(passenger_weights.hypothesis(training_passenger0).unwrap(), 1_f64.div(93_f64.exp().add(1_f64))),
        }
        ;
    }
    
    #[test]
    fn when_new_passenger_weights_and_training_passenger_and_gradient_descent_update_then_get_hypothesis() {
        // Initialize weights
        let mut passenger_weights = PassengerWeights::new();
        let training_passenger = TrainingPassenger::new(
            1_u64,
            Survived::Yes,
            PassengerClass::First,
            "Lewis Webb".to_string(),
            Sex::Male,
            25.33_f64,
            3_usize,
            2_usize,
            "Golden Ticket".to_string(),
            45.67_f64,
            "1".to_string(),
            PortOfEmbarkation::Southampton
        );
        let mut training_passengers = Vec::new();
        training_passengers.push(training_passenger);
        passenger_weights.gradient_descent_update(&(-1_f64), &training_passengers).unwrap();
        match training_passengers.get(0) {
            None => panic!("tests::when_new_passenger_weights_and_training_passenger_and_gradient_descent_update_then_get_hypothesis could not find item in vec"),
            Some(training_passenger0) => assert_abs_diff_eq!(passenger_weights.hypothesis(training_passenger0).unwrap(), 1_f64.div((83_f64.add((1_f64 - 1_f64.div(83_f64.exp().add(1_f64))).mul(10_f64))).exp().add(1_f64))),
        }
    }
}