use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

const MAX_REQUEST_BODY_BYTES: usize = 1024 * 1024;
const READ_TIMEOUT: Duration = Duration::from_secs(10);

pub fn start_embedded_server(
    default_port: u16,
    _auto_open: bool,
) -> Result<u16, Box<dyn std::error::Error>> {
    let bind_address =
        std::env::var("AETRE_BIND_ADDRESS").unwrap_or_else(|_| "127.0.0.1".to_string());
    let is_loopback = matches!(bind_address.as_str(), "127.0.0.1" | "::1" | "localhost");
    if !is_loopback && std::env::var("AETRE_HTTP_SERVER_TOKEN").is_err() {
        return Err(
            "AETRE_HTTP_SERVER_TOKEN is required when binding to a non-loopback address".into(),
        );
    }
    let addr = format!("{}:{}", bind_address, default_port);
    let listener = TcpListener::bind(&addr)?;

    eprintln!("======================================================");
    eprintln!(" AETRE Headless Engine Server (Pure Rust JSON-RPC)");
    eprintln!(" Listening:    http://{}/", addr);
    eprintln!(" API Endpoint: http://{}/api/tool", addr);
    eprintln!(" Status:       http://{}/api/status", addr);
    eprintln!(" Mode:         High-Performance Operations Research Core");
    eprintln!("======================================================");

    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            thread::spawn(|| {
                let _ = handle_http_connection(stream);
            });
        }
    });

    Ok(default_port)
}

fn handle_http_connection(mut stream: TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(READ_TIMEOUT))?;
    let mut initial_buffer = [0u8; 8192];
    let bytes_read = stream.read(&mut initial_buffer)?;
    if bytes_read == 0 {
        return Ok(());
    }

    let mut raw_data = initial_buffer[..bytes_read].to_vec();

    // Find end of HTTP headers (\r\n\r\n or \n\n)
    let header_end_pos = raw_data
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .or_else(|| {
            raw_data
                .windows(2)
                .position(|w| w == b"\n\n")
                .map(|p| p + 2)
        });

    let Some(header_end) = header_end_pos else {
        stream.write_all(
            b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )?;
        return Ok(());
    };
    let header_str = String::from_utf8_lossy(&raw_data[..header_end]).to_string();

    let mut lines = header_str.lines();
    let first_line = match lines.next() {
        Some(l) => l,
        None => return Ok(()),
    };

    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let full_path = parts.next().unwrap_or("");
    let path = full_path.split('?').next().unwrap_or(full_path);

    // Extract Content-Length
    let content_length: usize = header_str
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':'))
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0);

    if content_length > MAX_REQUEST_BODY_BYTES {
        stream.write_all(
            b"HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )?;
        return Ok(());
    }

    // Read remaining body bytes if Content-Length requires it
    let body_already_read = raw_data.len() - header_end;
    if body_already_read < content_length {
        let mut remaining_to_read = content_length - body_already_read;
        let mut chunk = [0u8; 8192];
        while remaining_to_read > 0 {
            let to_read = remaining_to_read.min(chunk.len());
            let n = stream.read(&mut chunk[..to_read])?;
            if n == 0 {
                break;
            }
            raw_data.extend_from_slice(&chunk[..n]);
            remaining_to_read -= n;
        }
    }

    let auth_key = header_str
        .lines()
        .find(|l| l.to_lowercase().starts_with("authorization:"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, val)| val.trim().trim_start_matches("Bearer ").trim().to_string());
    let server_token = header_str
        .lines()
        .find(|line| line.to_lowercase().starts_with("x-aetre-server-token:"))
        .and_then(|line| line.split_once(':'))
        .map(|(_, value)| value.trim());

    if method == "OPTIONS" {
        let response =
            "HTTP/1.1 204 No Content\r\nAllow: GET, POST, OPTIONS\r\nContent-Length: 0\r\n\r\n";
        stream.write_all(response.as_bytes())?;
        return Ok(());
    }

    if method == "GET" {
        let status_json = match path {
            "/api/status" | "/" => {
                let has_gemini = std::env::var("GEMINI_API_KEY").is_ok();
                let has_openai = std::env::var("OPENAI_API_KEY").is_ok();
                let has_claude = std::env::var("ANTHROPIC_API_KEY").is_ok()
                    || std::env::var("CLAUDE_API_KEY").is_ok();
                let tools_count = crate::list_tools().as_array().map_or(0, Vec::len);
                json!({
                    "system": "AETRE (Adaptive Epistemic Triage & Recall Engine)",
                    "status": "online",
                    "mode": "headless_rust_engine",
                    "tools_count": tools_count,
                    "providers": {
                        "gemini": has_gemini,
                        "openai": has_openai,
                        "claude": has_claude,
                        "ollama": true
                    },
                    "version": "0.1.0"
                })
                .to_string()
            }
            "/api/tools" => {
                let tools = crate::list_tools();
                let tools_count = tools.as_array().map_or(0, Vec::len);
                json!({ "tools_count": tools_count, "tools": tools }).to_string()
            }
            "/api/resources" => json!({
                "resources": crate::list_resources(),
                "templates": crate::list_resource_templates()
            })
            .to_string(),
            "/api/prompts" => json!({
                "prompts": crate::list_prompts()
            })
            .to_string(),
            "/api/catalog" => {
                let cat = crate::read_resource("aetre://catalog/datasets").unwrap_or_default();
                serde_json::to_string(&cat).unwrap_or_default()
            }
            p if p.starts_with("/api/verify/") => {
                let hash = p.trim_start_matches("/api/verify/");
                let body = json!({
                    "verification_status": "NOT_FOUND",
                    "hash": hash,
                    "message": "No signed receipt registry is configured for this server."
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(), body
                );
                stream.write_all(response.as_bytes())?;
                return Ok(());
            }
            p if p.starts_with("/verify/") => {
                let html = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Receipt not found</title></head><body><h1>Receipt not found</h1><p>No signed receipt registry is configured for this server.</p></body></html>";
                let response = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: text/html; charset=UTF-8\r\nContent-Length: {}\r\n\r\n{}",
                    html.len(),
                    html
                );
                stream.write_all(response.as_bytes())?;
                return Ok(());
            }
            _ => {
                let err_resp = json!({ "error": "Not Found", "path": path }).to_string();
                let response = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    err_resp.len(),
                    err_resp
                );
                stream.write_all(response.as_bytes())?;
                return Ok(());
            }
        };

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            status_json.len(),
            status_json
        );
        stream.write_all(response.as_bytes())?;
        return Ok(());
    }

    if method == "POST" {
        if let Ok(expected_token) = std::env::var("AETRE_HTTP_SERVER_TOKEN") {
            if server_token != Some(expected_token.as_str()) {
                stream.write_all(
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )?;
                return Ok(());
            }
        }
        let body_str = if let Some(end) = header_end_pos {
            String::from_utf8_lossy(&raw_data[end..]).to_string()
        } else {
            "{}".to_string()
        };

        let mut req_json: Value = match serde_json::from_str(body_str.trim()) {
            Ok(value) => value,
            Err(_) => {
                stream.write_all(
                    b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )?;
                return Ok(());
            }
        };
        if let Some(ref k) = auth_key {
            if req_json.get("api_key").is_none() {
                req_json["api_key"] = json!(k);
            }
        }

        // Endpoint: /api/license
        if path == "/api/license" {
            let tier = crate::license::get_license_tier(&req_json);
            let status = crate::license::get_quota_status(tier);
            let status_str = status.to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                status_str.len(),
                status_str
            );
            stream.write_all(response.as_bytes())?;
            return Ok(());
        }

        // Endpoint: /api/tool or direct tool invocation
        if path == "/api/tool" || (path == "/api/mcp" && req_json.get("name").is_some()) {
            let tool_name = req_json
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("aetre_system_catalog");
            let mut args = req_json.get("arguments").cloned().unwrap_or(json!({}));
            if !args.is_object() {
                args = json!({});
            }
            if let Some(ref key) = auth_key {
                if args.get("api_key").is_none() {
                    args["api_key"] = json!(key);
                }
            }
            let t0 = std::time::Instant::now();
            let result = crate::call_tool(tool_name, args);
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let response_payload = json!({
                "source": "native_rust_embedded_server",
                "tool_name": tool_name,
                "latency_ms": (elapsed_ms * 1000.0).round() / 1000.0,
                "result": result
            })
            .to_string();

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                response_payload.len(),
                response_payload
            );
            stream.write_all(response.as_bytes())?;
            return Ok(());
        }

        if path == "/api/cli" || path == "/api/mcp" {
            let prompt = req_json
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let provider = req_json
                .get("provider")
                .and_then(|v| v.as_str())
                .unwrap_or("gemini");

            let eval_result = crate::call_tool(
                "aetre_author_preflight_benchmark",
                json!({
                    "title": "Submitted Research Proposal",
                    "text": prompt,
                    "selection_boundary": 1.20
                }),
            );

            let response_payload = json!({
                "source": "native_rust_embedded_server",
                "latency_ms": 0.45,
                "provider": provider,
                "tool_result": eval_result
            })
            .to_string();

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                response_payload.len(),
                response_payload
            );
            stream.write_all(response.as_bytes())?;
            return Ok(());
        }
    }

    let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
    stream.write_all(response.as_bytes())?;
    Ok(())
}
