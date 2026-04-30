use axum::Router;

fn build_app() -> Router {
    Router::new()
}

#[tokio::main]
async fn main() {
    let _app = build_app();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_builds() {
        let _ = build_app();
    }
}
