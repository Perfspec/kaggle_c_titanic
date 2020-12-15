use std::ops::{Add, Mul, Div};

#[cfg(test)]
mod tests;

#[derive(Debug, PartialEq)]
pub enum BinaryClass {
	Yes,
	No
}

#[derive(Debug, PartialEq)]
pub struct Outcome {
	pub record_id: u64,
	pub prediction: BinaryClass
}

pub trait LogisticBinaryClassificationTestable {
	type Weights;
	
	fn hypothesis(self: &Self, weights: &Self::Weights) -> Result<f64, String>;
	
	fn get_record_id(self: &Self) -> &u64;
	
	fn logistic(val: f64) -> f64 {
		1_f64.div(val.exp().add(1_f64))
	}
}

pub trait LogisticBinaryClassificationTrainable: LogisticBinaryClassificationTestable {
	fn answer(self: &Self) -> BinaryClass;
	
	fn update_weights(self: &Self, diff: &f64, weights: &mut Self::Weights) -> Result<(), String>;
	
	fn diff_hypothesis(self: &Self, weights: &Self::Weights) -> Result<f64, String> {
        match self.answer() {
            BinaryClass::Yes => {
                match self.hypothesis(weights) {
                    Ok(hypothesis) => Ok(1_f64 - hypothesis),
                    Err(error) => Err(error),
                }
            },
            BinaryClass::No => {
                self.hypothesis(weights)
            },
        }
    }
	
	fn cost(self: &Self, weights: &Self::Weights) -> Result<f64, String> {
        match self.answer() {
            BinaryClass::Yes => {
                match self.hypothesis(weights) {
                    Ok(hypothesis) => Ok(-(hypothesis.ln())),
                    Err(error) => Err(error),
                }
            },
            BinaryClass::No => {
                match self.hypothesis(weights) {
                    Ok(hypothesis) => Ok(-((1_f64 - hypothesis).ln())),
                    Err(error) => Err(error)
                }
            },
        }
    }
}


	
pub fn quick_convert(num: &usize) -> f64 {
	let mut result = 0_f64;
	for _number in 0..*num {
		result = result.add(1_f64);
	}
	result
}

fn avg_cost<W, T>(training_records: &Vec<T>, weights: &W) -> Result<f64, String>
where
	T: LogisticBinaryClassificationTrainable<Weights = W>
{
	let mut sum = 0_f64;
	let mut counter = 0_f64;
	
	for record in training_records {
		match record.cost(&weights) {
			Ok(cost) => {
				sum = sum.add(cost);
				counter = counter.add(1_f64);
			},
			Err(e) => {
				let record_id = record.get_record_id();
				let message = format!("LogisticBinaryClassificationProblem::avg_cost was unable to calculate cost for record_id: {}. {}", record_id, e);
				return Err(message)
			},
		}
	}
	
	if counter.eq(&0_f64) {
		let message = "LogisticBinaryClassificationProblem::avg_cost counter is a denominator and was zero".to_string();
		return Err(message)
	}
	
	Ok(sum.div(counter))
}

fn gradient_descent_update<W, T>(training_records: &Vec<T>, weights: &mut W, learning_rate: &f64) -> Result<(), String>
where
	W: std::clone::Clone,
	T: LogisticBinaryClassificationTrainable<Weights = W>,
{
	let trainable_weights = weights.clone();
	for record in training_records {
		match record.diff_hypothesis(&trainable_weights) {
			Ok(diff) => {
				record.update_weights(&(diff.mul(-learning_rate)), weights)?;
			},
			Err(error) => return Err(error),
		}
	}
	Ok(())
}

pub fn predict<W, R>(weights: &W, record: &R) -> Result<Outcome, String>
where
	R: LogisticBinaryClassificationTestable<Weights = W>,
{
	match record.hypothesis(weights) {
		Ok(hypothesis) => {
			if hypothesis > 0.5_f64 {
				Ok(Outcome {
					record_id: *(record.get_record_id()),
					prediction: BinaryClass::Yes
				})
			} else {
				Ok(Outcome {
					record_id: *(record.get_record_id()),
					prediction: BinaryClass::No
				})
			}
		},
		Err(error) => Err(error),
	}
}

fn predict_batch<W, R>(weights: &W, records: &Vec<R>) -> Result<Vec<Outcome>, String>
where
	R: LogisticBinaryClassificationTestable<Weights = W>,
{
	let mut outcome_vec = Vec::new();
	
	for record in records {
		match predict(weights, record) {
			Ok(outcome) => {
				outcome_vec.push(outcome);
			},
			Err(error) => return Err(error),
		}
	}
	
	Ok(outcome_vec)
}

pub fn solve<W, T>(training_records: &Vec<T>, weights: &mut W, mut learning_rate: f64, tolerance: &f64) -> Result<(), String>
where
	W: std::clone::Clone + std::fmt::Debug,
	T: LogisticBinaryClassificationTrainable<Weights = W>,
{
	match avg_cost(training_records, weights) {
		Ok(initial_cost) => {
			let mut current_avg_cost = initial_cost;
			let mut num_iterations = 0_u64;
			println!("LogisticBinaryClassificationProblem::solve At iteration {}, the avg_cost is {}", &num_iterations, &current_avg_cost);
			
			while current_avg_cost.gt(tolerance) {
				match gradient_descent_update(training_records, weights, &learning_rate) {
					Ok(_) => {
						num_iterations = num_iterations.add(1_u64);
						match avg_cost(training_records, weights) {
							Ok(new_avg_cost) => {
								if current_avg_cost.lt(&new_avg_cost) {
									learning_rate = learning_rate.div(100_f64);
									println!("LogisticBinaryClassificationProblem::solve Learning rate divided by 10 at iteration {}. New learning_rate: {}", &num_iterations, &learning_rate);
								};
								current_avg_cost = new_avg_cost;
								println!("LogisticBinaryClassificationProblem::solve At iteration {}, the avg_cost is {}", &num_iterations, &current_avg_cost);
							},
							Err(error) => return Err(error),
						}
					},
					Err(error) => return Err(error),
				}
			}
			println!("LogisticBinaryClassificationProblem::solve Tolerable Cost Achieved: weights={:#?}", &weights);
			Ok(())
		},
		Err(error) => Err(error),
	}
}