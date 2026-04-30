use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "llamas-cli")]
struct Args {
    #[arg(long, default_value = "hello")]
    prompt: String,
}

fn greeting(prompt: &str) -> String {
    format!("llamas-cli: {prompt}")
}

fn main() {
    let args = Args::parse();
    println!("{}", greeting(&args.prompt));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_uses_prompt() {
        assert_eq!(greeting("test"), "llamas-cli: test");
    }
}
