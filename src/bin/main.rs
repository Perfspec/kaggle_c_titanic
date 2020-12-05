use std::process;
use kaggle_c_titanic::Config;

fn main() {
	// Argument 1: Any String
	// Argument 2: Learning Rate of Gradient Descent
	// Argument 3: Tolerance of Gradient Descent
	// Argument 4: Path to the Training Data
	// Argument 5: Path to the Test Data
	// Argument 6: Path where the Output Data will be created
    let args: Vec<String> = vec![
		"This vector can be loaded using std::env::args().collect() too.".to_string(),
		"0.1".to_string(),
		"0.0001".to_string(),
		"data/train.csv".to_string(),
		"data/test.csv".to_string(),
		"output.csv".to_string()];

    let mut config = Config::new(&args).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });
	
	if let Err(e) = kaggle_c_titanic::run(&mut config) {
        eprintln!("Application error: {}", e);

        process::exit(1);
    }
}