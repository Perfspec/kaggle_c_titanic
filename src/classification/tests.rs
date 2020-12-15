use super::*;

enum FootbalFan {
	Yes,
	No
}

struct TrainingData {
	pub party_id: u64,
	pub age: f64,
	pub weight: f64,
	pub footbal_fan: FootbalFan
}

struct TestData {
	pub party_id: u64,
	pub age: f64,
	pub weight: f64
}

struct FootbalFanWeights {
	pub age: f64,
	pub weight: f64
}

impl Clone for FootbalFanWeights {
	fn clone(&self) -> Self {
		FootbalFanWeights {
			age: self.age.clone(),
			weight: self.weight.clone()
		}
	}
}

impl LogisticBinaryClassificationTestable for TrainingData {
	type Weights = FootbalFanWeights;
	
	fn hypothesis(self: &Self, weights: &Self::Weights) -> Result<f64, String> {
		Ok(Self::logistic(self.age.mul(weights.age).add(self.weight.mul(weights.weight))))
	}
	
	fn get_record_id(self: &Self) -> &u64 {
		&self.party_id
	}
}

impl LogisticBinaryClassificationTrainable for TrainingData {
	fn answer(self: &Self) -> BinaryClass {
		match self.footbal_fan {
			FootbalFan::Yes => BinaryClass::Yes,
			FootbalFan::No => BinaryClass::No
		}
	}
	
	fn update_weights(self: &Self, diff: &f64, weights: &mut Self::Weights) -> Result<(), String> {
		weights.age = weights.age.add(diff);
		weights.weight = weights.weight.add(diff);
		Ok(())
	}
}

impl LogisticBinaryClassificationTestable for TestData {
	type Weights = FootbalFanWeights;
	
	fn hypothesis(self: &Self, weights: &Self::Weights) -> Result<f64, String> {
		Ok(Self::logistic(self.age.mul(weights.age).add(self.weight.mul(weights.weight))))
	}
	
	fn get_record_id(self: &Self) -> &u64 {
		&self.party_id
	}
}



#[test]
fn when_new_training_data_and_weights_then_hypothesis_correct() {
	let training_data = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::Yes
	};
	
	let footbal_fan_weights = FootbalFanWeights {
		age: 1_f64,
		weight: 1_f64
	};
	
	assert_abs_diff_eq!(training_data.hypothesis(&footbal_fan_weights).unwrap(), 1_f64.div(2_f64.exp().add(1_f64)));
}

#[test]
fn when_new_training_data_and_weights_then_record_id_correct() {
	let training_data = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::Yes
	};
	
	assert_abs_diff_eq!(*training_data.get_record_id(), 1_u64);
}

#[test]
fn when_new_test_data_and_weights_then_hypothesis_correct() {
	let test_data = TestData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64
	};
	
	let footbal_fan_weights = FootbalFanWeights {
		age: 1_f64,
		weight: 1_f64
	};
	
	assert_abs_diff_eq!(test_data.hypothesis(&footbal_fan_weights).unwrap(), 1_f64.div(2_f64.exp().add(1_f64)));
}

#[test]
fn when_new_test_data_and_weights_then_record_id_correct() {
	let test_data = TestData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64
	};
	
	assert_abs_diff_eq!(*test_data.get_record_id(), 1_u64);
}

#[test]
fn when_new_training_data_and_weights_and_weights_updated_then_hypothesis_correct() {
	let training_data = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::Yes
	};
	
	let mut footbal_fan_weights = FootbalFanWeights {
		age: 1_f64,
		weight: 1_f64
	};
	
	training_data.update_weights(&1_f64, &mut footbal_fan_weights).unwrap();
	
	assert_abs_diff_eq!(training_data.hypothesis(&footbal_fan_weights).unwrap(), 1_f64.div(4_f64.exp().add(1_f64)));
}

#[test]
fn when_new_training_data_then_answer_correct() {
	let training_data = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::Yes
	};
	
	assert_eq!(training_data.answer(), BinaryClass::Yes);
}

#[test]
fn when_new_training_data_and_weights_then_diff_hypothesis_correct() {
	let training_data1 = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::Yes
	};
	
	let training_data2 = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::No
	};
	
	let footbal_fan_weights = FootbalFanWeights {
		age: 1_f64,
		weight: 1_f64
	};
	
	assert_abs_diff_eq!(training_data1.diff_hypothesis(&footbal_fan_weights).unwrap(), 1_f64 - 1_f64.div(2_f64.exp().add(1_f64)));
	assert_abs_diff_eq!(training_data2.diff_hypothesis(&footbal_fan_weights).unwrap(), 1_f64.div(2_f64.exp().add(1_f64)));
}

#[test]
fn when_new_training_data_and_weights_then_cost_correct() {
	let training_data1 = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::Yes
	};
	
	let training_data2 = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::No
	};
	
	let footbal_fan_weights = FootbalFanWeights {
		age: 1_f64,
		weight: 1_f64
	};
	
	assert_abs_diff_eq!(training_data1.cost(&footbal_fan_weights).unwrap(), - 1_f64.div(2_f64.exp().add(1_f64)).ln());
	assert_abs_diff_eq!(training_data2.cost(&footbal_fan_weights).unwrap(), - (1_f64 - 1_f64.div(2_f64.exp().add(1_f64))).ln());
}

#[test]
fn when_new_training_data_and_weights_then_avg_cost_correct() {
	let training_data1 = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::Yes
	};
	
	let training_data2 = TrainingData {
		party_id: 1_u64,
		age: 1_f64,
		weight: 1_f64,
		footbal_fan: FootbalFan::No
	};
	
	let footbal_fan_weights = FootbalFanWeights {
		age: 1_f64,
		weight: 1_f64
	};
	
	let mut training_data = Vec::new();
	training_data.push(training_data1);
	training_data.push(training_data2);
	
	assert_abs_diff_eq!(avg_cost(&training_data, &footbal_fan_weights).unwrap(), ((- 1_f64.div(2_f64.exp().add(1_f64)).ln()) + (- (1_f64 - 1_f64.div(2_f64.exp().add(1_f64))).ln())).div(2_f64));
}