use std::process;
use kaggle_c_titanic::Config;

fn main() {
	// Argument 1: Any String
	// Argument 2: Path to the Training Data
	// Argument 3: Path to the Training Data
	// Argument 4: Path where the Output Data will be created
    let args: Vec<String> = vec![
		"This vector can be loaded using std::env::args().collect() too.".to_string(),
		"data/train.csv".to_string(),
		"data/test.csv".to_string(),
		"output.csv".to_string()];

    let config = Config::new(&args).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });
	
	if let Err(e) = kaggle_c_titanic::run(config) {
        eprintln!("Application error: {}", e);

        process::exit(1);
    }
}