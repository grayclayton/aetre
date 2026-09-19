#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layer {
    All,
    Macro,
    Micro,
}

static LAYER: std::sync::OnceLock<Layer> = std::sync::OnceLock::new();

pub fn set_layer(layer: Layer) {
    let _ = LAYER.set(layer);
}

pub fn active_layer() -> Layer {
    *LAYER.get().unwrap_or(&Layer::All)
}

pub fn layer_from_args(args: &[String]) -> Result<Layer, String> {
    let mut requested: Option<&str> = None;
    for (i, arg) in args.iter().enumerate() {
        if let Some(value) = arg.strip_prefix("--layer=") {
            requested = Some(value);
        } else if arg == "--layer" {
            requested = args.get(i + 1).map(String::as_str);
        }
    }
    match requested {
        None => Ok(Layer::All),
        Some("all") => Ok(Layer::All),
        Some("macro") => Ok(Layer::Macro),
        Some("micro") => Ok(Layer::Micro),
        Some(other) => Err(format!(
            "unknown --layer '{other}': expected macro, micro or all"
        )),
    }
}

pub fn tool_in_layer(name: &str, layer: Layer) -> bool {
    match layer {
        Layer::All => true,
        Layer::Macro => !name.starts_with("governed_"),
        Layer::Micro => !name.starts_with("aetre_"),
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;
    use crate::schemas::all_tools;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn layer_parses_both_spellings_and_defaults_to_all() {
        assert_eq!(layer_from_args(&args(&["aetre-mcp"])).unwrap(), Layer::All);
        assert_eq!(
            layer_from_args(&args(&["aetre-mcp", "--layer=micro"])).unwrap(),
            Layer::Micro
        );
        assert_eq!(
            layer_from_args(&args(&["aetre-mcp", "--layer", "macro"])).unwrap(),
            Layer::Macro
        );
        assert!(layer_from_args(&args(&["aetre-mcp", "--layer=sideways"])).is_err());
    }

    #[test]
    fn layers_partition_the_tools_and_leave_unknown_names_alone() {
        let names: Vec<String> = all_tools()
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();

        let macro_count = names
            .iter()
            .filter(|n| tool_in_layer(n, Layer::Macro))
            .count();
        let micro_count = names
            .iter()
            .filter(|n| tool_in_layer(n, Layer::Micro))
            .count();
        assert_eq!(macro_count + micro_count, names.len());
        assert_eq!(macro_count, 22);
        assert_eq!(micro_count, 10);

        // An unknown tool must stay unknown rather than become "not exposed".
        assert!(tool_in_layer("does_not_exist", Layer::Macro));
        assert!(tool_in_layer("does_not_exist", Layer::Micro));
    }
}
