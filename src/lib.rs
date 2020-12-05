use csv::Reader;
use serde::Deserialize;
use std::ops::{Add, Mul, Div};

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

pub fn run(config: Config) -> Result<(), String> {
    //Read training_data into vector of training_passengers, which will be reused many times.
    match Reader::from_path(config.training_data_filename) {
        Ok(training_data) => {
            let mut training_passengers = Vec::new();
    
            for result in training_data.deserialize() {
                match result {
                    Ok(result) => {
                        let mut training_passenger: TrainingPassenger = result;
                        training_passengers.push(training_passenger);
                    },
                    Err(_) => return Err("Failed to deserialize TrainingPassenger".to_string()),
                }
                
            }
            
            // Initialize weights
            let mut passenger_weights = PassengerWeights::new();
            
            let mut avg_cost = 0_f64;
            let mut num_iterations = 0_u64;

            match passenger_weights.avg_cost(&training_passengers) {
                Ok(num) => {
                    avg_cost = num;
                    println!("At iteration {}, the avg_cost is {}", num_iterations, avg_cost);
                    
                    while avg_cost.gt(&config.tolerance) {
                        match passenger_weights.gradient_descent_update(&config.learning_rate, &training_passengers) {
                            Ok(_) => {
                                num_iterations.add(1_u64);
                                match passenger_weights.avg_cost(&training_passengers) {
                                    Ok(num) => {
                                        if avg_cost.lt(&num) {
                                            config.learning_rate.div(10_f64);
                                            println!("Learning rate divided by 10 at iteration {}. New learning_rate: {}", &num_iterations, &config.learning_rate);
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
                },
                Err(error) => return Err(error),
            }
                
            Ok(())
        },
        Err(_) => {
            let message = format!("Failed to read from {}", config.training_data_filename);
            return Err(message)
        },
    }
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
    
    pub fn get_survived_bit(&self) -> f64 {
        match self.survived {
            Survived::Yes => 1_f64,
            Survived::No => 0_f64,
        }
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
        
        //Optional floats and integers only have one param instantiated for when Optional matches None.
        //When Optional matches Some(value), then extra capacity will be reserved at runtime.
        let mut age = Vec::new();
        age.push(1_f64);
        
        let mut siblings_spouses = Vec::new();
        siblings_spouses.push(1_f64);
        
        let mut parents_children = Vec::new();
        parents_children.push(1_f64);
        
        let mut fare = Vec::new();
        fare.push(1_f64);
        
        //Optional strings will only be differentiate by Some or None for now.
        //In future versions, the strings may be categorized, so capacity may be reserved at runtime or a fixed number of categories. TBD
        let mut name = Vec::with_capacity(2);
        name.push(1_f64);
        name.push(1_f64);
        
        let mut ticket_id = Vec::with_capacity(2);
        ticket_id.push(1_f64);
        ticket_id.push(1_f64);
        
        let mut cabin_id = Vec::with_capacity(2);
        cabin_id.push(1_f64);
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
    
    pub fn hypothesis(&mut self, training_passenger: &TrainingPassenger) -> Result<f64, String> {
        let mut weighted_sum = 0_f64;
        
        weighted_sum.add(self.bias);
        
        match training_passenger.get_name() {
            None => {
                match self.name.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(name) => {
                match self.name.get(1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match training_passenger.get_age() {
            None => {
                match self.age.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(age) => {
                let age_usize = unsafe { age.to_int_unchecked::<usize>() };
                if (self.age.len()).lt(&age_usize.add(1_usize)) {
                    self.age.resize(age_usize.add(1_usize), 1_f64);
                }
                match self.age.get(age_usize) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: age weight {} was unreachable for passenger {}", &age_usize, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight.mul(age.trunc()));
                    },
                }
            },
        }
        
        match training_passenger.get_siblings_spouses() {
            None => {
                match self.siblings_spouses.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(siblings_spouses) => {
                if (self.siblings_spouses.len()).lt(&siblings_spouses.add(1_usize)) {
                    self.siblings_spouses.resize(siblings_spouses.add(1_usize), 1_f64);
                }
                match self.siblings_spouses.get(*siblings_spouses) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: siblings_spouses weight {} was unreachable for passenger {}", siblings_spouses, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        let cast: f64 = siblings_spouses.into();
                        weighted_sum.add(weight.mul(cast));
                    },
                }
            },
        }
        
        match training_passenger.get_parents_children() {
            None => {
                match self.parents_children.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(parents_children) => {
                if (self.parents_children.len()).lt(&parents_children.add(1_usize)) {
                    self.parents_children.resize(parents_children.add(1_usize), 1_f64);
                }
                match self.parents_children.get(*parents_children) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: parents_children weight {} was unreachable for passenger {}", parents_children, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        let cast: f64 = parents_children.into();
                        weighted_sum.add(weight.mul(cast));
                    },
                }
            },
        }
        
        match training_passenger.get_fare() {
            None => {
                match self.fare.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(fare) => {
                let fare_usize = unsafe { fare.to_int_unchecked::<usize>() };
                if (self.fare.len()).lt(&fare_usize.add(1_usize)) {
                    self.fare.resize(fare_usize.add(1_usize), 1_f64);
                }
                match self.fare.get(fare_usize) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: fare weight {} was unreachable for passenger {}", fare_usize, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight.mul(fare.trunc()));
                    },
                }
            },
        }
        
        match training_passenger.get_ticket_id() {
            None => {
                match self.ticket_id.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(ticket_id) => {
                match self.ticket_id.get(1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match training_passenger.get_cabin_id() {
            None => {
                match self.cabin_id.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(cabin_id) => {
                match self.cabin_id.get(1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
        }
        
        match training_passenger.get_passenger_class() {
            None => {
                match self.passenger_class.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
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
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match self.passenger_class.get(2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match self.passenger_class.get(3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }
        
        match training_passenger.get_sex() {
            None => {
                match self.sex.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: sex weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
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
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                    Sex::Male => {
                        match self.sex.get(2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: sex weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }
        
        match training_passenger.get_port_of_embarkation() {
            None => {
                match self.port_of_embarkation.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weighted_sum.add(weight);
                    },
                }
            },
            Some(port_of_embarkation) => {
                match port_of_embarkation {
                    PortOfEmbarkation::Cherbourg => {
                        match self.port_of_embarkation.get(1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match self.port_of_embarkation.get(2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match self.port_of_embarkation.get(3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::hypothesis: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weighted_sum.add(weight);
                            },
                        }
                    },
                }
            },
        }

        let mut hypothesis = 1_f64;

        Ok(hypothesis.div(weighted_sum.exp().add(1_f64)))
    }
    
    pub fn cost(&mut self, training_passenger: &TrainingPassenger) -> Result<f64, String> {
        let mut cost = 0_f64;
        
        match training_passenger.get_survived() {
            Survived::Yes => {
                match self.hypothesis(training_passenger) {
                    Ok(hypothesis) => {
                        cost.add(hypothesis.ln());
                    },
                    Err(error) => return Err(error),
                }
            },
            Survived::No => {
                match self.hypothesis(training_passenger) {
                    Ok(hypothesis) => {
                        cost.add((1_f64 - hypothesis).ln());
                    },
                    Err(error) => return Err(error),
                }
            },
        }
        Ok(-cost)
    }
    
    pub fn avg_cost(&mut self, training_passengers: &Vec<TrainingPassenger>) -> Result<f64, String> {
        let mut sum = 0_f64;
        let mut counter = 0_f64;
        
        for training_passenger in *training_passengers {
            match self.cost(&training_passenger) {
                Ok(cost) => {
                    sum.add(cost);
                },
                Err(e) => {
                    let passenger_id = training_passenger.get_passenger_id();
                    let message = format!("PassengerWeights::avg_cost was unable to calculate cost for passenger_id: {}. {}", passenger_id, e);
                    return Err(message)
                },
            }
            counter.add(1_f64);
        }
        
        if counter.eq(&0_f64) {
            let message = "PassengerWeights::avg_cost counter is a denominator and was zero".to_string();
            return Err(message)
        }
        
        let avg = sum.div(counter);
        Ok(avg)
    }
    
    pub fn gradient_descent_update(&mut self, learning_rate: &f64, training_passengers: &Vec<TrainingPassenger>) -> Result<(), String> {
        for training_passenger in *training_passengers {
            match self.diff_hypothesis(&training_passenger) {
                Ok(diff) => {
                    self.add(&(diff.mul(training_passenger.get_survived_bit()).mul(-1_f64).mul(learning_rate)), &training_passenger)?;
                },
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    
    fn diff_hypothesis(&mut self, training_passenger: &TrainingPassenger) -> Result<f64, String> {
        let mut diff = 0_f64;
        
        match *training_passenger.get_survived() {
            Survived::Yes => {
                match self.hypothesis(training_passenger) {
                    Ok(hypothesis) => {
                        diff.add(hypothesis.add(-1_f64));
                    },
                    Err(error) => return Err(error),
                }
            },
            Survived::No => {
                match self.hypothesis(training_passenger) {
                    Ok(hypothesis) => {
                        diff.add(hypothesis);
                    },
                    Err(error) => return Err(error),
                }
            },
        }
        Ok(-diff)
    }
    
    fn add(&mut self, diff: &f64, training_passenger: &TrainingPassenger) -> Result<(), String> {
        self.bias.add(diff);
        
        match training_passenger.get_name() {
            None => {
                match self.name.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: name weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(name) => {
                match self.name.get(1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: name weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
        }
        
        match training_passenger.get_age() {
            None => {
                match self.age.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: age weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(age) => {
                let age_usize = unsafe { age.to_int_unchecked::<usize>() };
                if (self.age.len()).lt(&age_usize.add(1_usize)) {
                    self.age.resize(age_usize.add(1_usize), 1_f64);
                }
                match self.age.get(age_usize) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: age weight {} was unreachable for passenger {}", age_usize, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight).mul(age.trunc()));
                    },
                }
            },
        }
        
        match training_passenger.get_siblings_spouses() {
            None => {
                match self.siblings_spouses.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: siblings_spouses weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(siblings_spouses) => {
                if (self.siblings_spouses.len()).lt(&siblings_spouses.add(1_usize)) {
                    self.siblings_spouses.resize(siblings_spouses.add(1_usize), 1_f64);
                }
                match self.siblings_spouses.get(*siblings_spouses) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: siblings_spouses weight {} was unreachable for passenger {}", siblings_spouses, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        let cast: f64 = siblings_spouses.into();
                        weight.add(diff.mul(weight).mul(cast));
                    },
                }
            },
        }
        
        match training_passenger.get_parents_children() {
            None => {
                match self.parents_children.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: parents_children weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(parents_children) => {
                if (self.parents_children.len()).lt(&parents_children.add(1_usize)) {
                    self.parents_children.resize(parents_children.add(1_usize), 1_f64);
                }
                match self.parents_children.get(*parents_children) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: parents_children weight {} was unreachable for passenger {}", parents_children, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        let cast: f64 = parents_children.into();
                        weight.add(diff.mul(weight).mul(cast));
                    },
                }
            },
        }
        
        match training_passenger.get_fare() {
            None => {
                match self.fare.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: fare weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(fare) => {
                let fare_usize = unsafe { fare.to_int_unchecked::<usize>() };
                if (self.fare.len()).lt(&fare_usize.add(1_usize)) {
                    self.fare.resize(fare_usize.add(1_usize), 1_f64);
                }
                match self.fare.get(fare_usize) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: fare weight {} was unreachable for passenger {}", fare_usize, passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight).mul(fare.trunc()));
                    },
                }
            },
        }
        
        match training_passenger.get_ticket_id() {
            None => {
                match self.ticket_id.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: ticket_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(ticket_id) => {
                match self.ticket_id.get(1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: ticket_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
        }
        
        match training_passenger.get_cabin_id() {
            None => {
                match self.cabin_id.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: cabin_id weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(cabin_id) => {
                match self.cabin_id.get(1) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: cabin_id weight 1 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
        }
        
        match training_passenger.get_passenger_class() {
            None => {
                match self.passenger_class.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: passenger_class weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(passenger_class) => {
                match passenger_class {
                    PassengerClass::First => {
                        match self.passenger_class.get(1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: passenger_class weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                    PassengerClass::Second => {
                        match self.passenger_class.get(2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: passenger_class weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                    PassengerClass::Third => {
                        match self.passenger_class.get(3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: passenger_class weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                }
            },
        }
        
        match training_passenger.get_sex() {
            None => {
                match self.sex.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: sex weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(sex) => {
                match sex {
                    Sex::Female => {
                        match self.sex.get(1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: sex weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                    Sex::Male => {
                        match self.sex.get(2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: sex weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                }
            },
        }
        
        match training_passenger.get_port_of_embarkation() {
            None => {
                match self.port_of_embarkation.get(0) {
                    None => {
                        let passenger_id = training_passenger.get_passenger_id();
                        let message = format!("PassengerWeights::add: port_of_embarkation weight 0 was unreachable for passenger {}", passenger_id);
                        return Err(message)
                    },
                    Some(weight) => {
                        weight.add(diff.mul(weight));
                    },
                }
            },
            Some(port_of_embarkation) => {
                match port_of_embarkation {
                    PortOfEmbarkation::Cherbourg => {
                        match self.port_of_embarkation.get(1) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: port_of_embarkation weight 1 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                    PortOfEmbarkation::Southampton => {
                        match self.port_of_embarkation.get(2) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: port_of_embarkation weight 2 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                    PortOfEmbarkation::Queenstown => {
                        match self.port_of_embarkation.get(3) {
                            None => {
                                let passenger_id = training_passenger.get_passenger_id();
                                let message = format!("PassengerWeights::add: port_of_embarkation weight 3 was unreachable for passenger {}", passenger_id);
                                return Err(message)
                            },
                            Some(weight) => {
                                weight.add(diff.mul(weight));
                            },
                        }
                    },
                }
            },
        }

        Ok(())
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
    //    let mut passengers = Vec::new();
    //    let passenger = TrainingPassenger {
    //        
    //    }
    //    passenger.push(passenger);
    //    passenger_weights.gradient_descent_update(&passengers).unwrap();
    //    
    //    assert_eq!(passenger_weights.get_bias(), 10.0);
    //}
}
