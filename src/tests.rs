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