fn quantize_name(model: &str) -> String {
    format!("{model}.q")
}

fn main() {
    println!("{}", quantize_name("model"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_name_suffix() {
        assert_eq!(quantize_name("weights"), "weights.q");
    }
}
