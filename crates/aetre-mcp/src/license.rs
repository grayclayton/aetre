use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use ed25519_compact::{KeyPair, PublicKey, Signature};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Deterministic 32-byte Root Public Key embedded directly in the AETRE binary for offline verification.
pub const ROOT_PUBLIC_KEY_BYTES: [u8; 32] = [
    0xa6, 0x6c, 0xb7, 0x28, 0x27, 0x1a, 0xc4, 0x7b, 0xb0, 0x46, 0x69, 0xfe, 0x7d, 0x02, 0x16, 0xba,
    0x29, 0xc1, 0xf7, 0x22, 0x94, 0x33, 0xf5, 0x08, 0x67, 0x10, 0x3a, 0xcb, 0xf6, 0x78, 0x41, 0xd3,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseTier {
    Community,
    Pro,
    Enterprise,
}

impl LicenseTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            LicenseTier::Community => "COMMUNITY_FREE",
            LicenseTier::Pro => "PRO_AUTHOR",
            LicenseTier::Enterprise => "VC_ENTERPRISE",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            LicenseTier::Community => "Community (Free Open-Core)",
            LicenseTier::Pro => "Pro Author ($19/check or $49/mo)",
            LicenseTier::Enterprise => "VC & Institutional Enterprise",
        }
    }
}

/// Cryptographically signed license claims payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LicenseClaims {
    pub sub: String,
    pub tier: LicenseTier,
    pub features: Vec<String>,
    pub iat: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_monthly_evals: Option<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LicenseError {
    InvalidFormat,
    InvalidBase64,
    InvalidJson,
    InvalidSignature,
    Expired,
}

impl std::fmt::Display for LicenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseError::InvalidFormat => write!(f, "Invalid license key format"),
            LicenseError::InvalidBase64 => write!(f, "Invalid base64 encoding in license token"),
            LicenseError::InvalidJson => write!(f, "Corrupted license claims JSON"),
            LicenseError::InvalidSignature => write!(f, "Cryptographic signature mismatch"),
            LicenseError::Expired => write!(f, "License key has expired"),
        }
    }
}

/// Generates a signed Ed25519 license key token (Issuer Utility).
#[allow(dead_code)]
pub fn generate_signed_license(
    key_pair: &KeyPair,
    claims: &LicenseClaims,
    is_live: bool,
) -> String {
    let payload_json = serde_json::to_vec(claims).expect("Claims serialization failed");
    let payload_b64 = URL_SAFE_NO_PAD.encode(&payload_json);
    let signature = key_pair.sk.sign(payload_b64.as_bytes(), None);
    let sig_b64 = URL_SAFE_NO_PAD.encode(signature.as_ref());

    let prefix = if is_live {
        "aetre_live_"
    } else {
        "aetre_test_"
    };
    format!("{}{}.{}", prefix, payload_b64, sig_b64)
}

/// Parses and cryptographically verifies an Ed25519 signed license token against a public key.
pub fn parse_and_verify_token(
    token: &str,
    public_key_bytes: &[u8],
) -> Result<LicenseClaims, LicenseError> {
    let token = token.trim();
    let body = if let Some(stripped) = token.strip_prefix("aetre_live_") {
        stripped
    } else if let Some(stripped) = token.strip_prefix("aetre_test_") {
        stripped
    } else {
        return Err(LicenseError::InvalidFormat);
    };

    let parts: Vec<&str> = body.split('.').collect();
    if parts.len() != 2 {
        return Err(LicenseError::InvalidFormat);
    }

    let payload_b64 = parts[0];
    let sig_b64 = parts[1];

    let sig_bytes = URL_SAFE_NO_PAD
        .decode(sig_b64)
        .map_err(|_| LicenseError::InvalidBase64)?;
    if sig_bytes.len() != 64 {
        return Err(LicenseError::InvalidSignature);
    }

    let signature =
        Signature::from_slice(&sig_bytes).map_err(|_| LicenseError::InvalidSignature)?;
    let public_key =
        PublicKey::from_slice(public_key_bytes).map_err(|_| LicenseError::InvalidSignature)?;

    // Verify signature over the payload_b64 string
    public_key
        .verify(payload_b64.as_bytes(), &signature)
        .map_err(|_| LicenseError::InvalidSignature)?;

    // Decode and deserialize payload
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| LicenseError::InvalidBase64)?;
    let claims: LicenseClaims =
        serde_json::from_slice(&payload_bytes).map_err(|_| LicenseError::InvalidJson)?;

    // Validate expiration
    if let Some(exp) = claims.exp {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now > exp {
            return Err(LicenseError::Expired);
        }
    }

    Ok(claims)
}

/// Reads any local license key saved in ~/.aetre/license.key.
pub fn get_local_license_key_file() -> Option<String> {
    let path = if let Ok(profile) = env::var("USERPROFILE") {
        PathBuf::from(profile).join(".aetre").join("license.key")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".aetre").join("license.key")
    } else {
        return None;
    };

    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// Saves a license key to ~/.aetre/license.key.
#[allow(dead_code)]
pub fn save_local_license_key(key: &str) -> std::io::Result<()> {
    let dir = if let Ok(profile) = env::var("USERPROFILE") {
        PathBuf::from(profile).join(".aetre")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".aetre")
    } else {
        env::temp_dir().join(".aetre")
    };

    fs::create_dir_all(&dir)?;
    let path = dir.join("license.key");
    fs::write(&path, key.trim())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// Detailed resolution result for status reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedLicense {
    pub tier: LicenseTier,
    pub source: String,
    pub is_verified_crypto: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,
}

/// Resolves the full license status from tool arguments, headers, env vars, or local config file.
/// Returns license status overview for aetre_system_catalog.
///
/// Every tool runs at every tier. A commercial license buys an exemption from
/// AGPL-3.0 copyleft for closed-source and hosted use, not extra tools, so
/// there is nothing here to lock.
pub fn get_quota_status(tier: LicenseTier) -> Value {
    json!({
        "active_tier": tier.as_str(),
        "tier_name": tier.display_name(),
        "tool_access": "All tools are available at every tier.",
        "commercial_license_covers": "Exemption from AGPL-3.0 copyleft for closed-source and hosted use.",
        "licensing": "https://www.lithiumeel.com/aetre"
    })
}

pub fn resolve_license(args: &Value) -> ResolvedLicense {
    resolve_license_with_key(args, &ROOT_PUBLIC_KEY_BYTES)
}

/// Resolves license status verifying against a specific public key (useful for test harnesses).
pub fn resolve_license_with_key(args: &Value, pub_key: &[u8]) -> ResolvedLicense {
    // 1. Check Tool Arguments
    let key_from_args = args
        .get("api_key")
        .or_else(|| args.get("key"))
        .or_else(|| args.get("license_key"))
        .or_else(|| args.get("token"))
        .and_then(|v| v.as_str());

    // 2. Check Environment Variables
    let key_from_env = env::var("AETRE_LICENSE_KEY")
        .or_else(|_| env::var("AETRE_API_KEY"))
        .ok();

    // 3. Check Local Config File (~/.aetre/license.key)
    let key_from_file = get_local_license_key_file();

    let (raw_key, source) = if let Some(k) = key_from_args {
        (k.to_string(), "tool_arguments")
    } else if let Some(k) = key_from_env {
        (k, "environment_variable")
    } else if let Some(k) = key_from_file {
        (k, "local_config_file (~/.aetre/license.key)")
    } else {
        ("".to_string(), "default_community")
    };

    let key = raw_key.trim();

    // Cryptographic Token Verification
    if key.starts_with("aetre_live_") || key.starts_with("aetre_test_") {
        if let Ok(claims) = parse_and_verify_token(key, pub_key) {
            return ResolvedLicense {
                tier: claims.tier,
                source: format!("{} [Ed25519 Cryptographically Verified]", source),
                is_verified_crypto: true,
                subject: Some(claims.sub),
                expires_at: claims.exp,
                features: Some(claims.features),
            };
        }
    }

    // Test-only aliases. Production and developer builds require signed tokens.
    #[cfg(test)]
    {
        let key_lower = key.to_lowercase();
        if key_lower.starts_with("aetre_ent_")
            || key_lower.contains("_ent_")
            || key_lower == "aetre_enterprise_key"
            || key_lower == "enterprise"
        {
            return ResolvedLicense {
                tier: LicenseTier::Enterprise,
                source: format!("{} [Enterprise Debug Alias]", source),
                is_verified_crypto: false,
                subject: Some("Developer / Demo Enterprise".to_string()),
                expires_at: None,
                features: Some(vec!["*".to_string()]),
            };
        } else if key_lower.starts_with("aetre_pro_")
            || key_lower.contains("_pro_")
            || key_lower == "aetre_pro_key"
            || key_lower == "pro"
        {
            return ResolvedLicense {
                tier: LicenseTier::Pro,
                source: format!("{} [Pro Debug Alias]", source),
                is_verified_crypto: false,
                subject: Some("Developer / Demo Pro".to_string()),
                expires_at: None,
                features: Some(vec!["author_preflight".to_string()]),
            };
        }
    }

    // Fallback: Community Free Tier
    ResolvedLicense {
        tier: LicenseTier::Community,
        source: "community_default".to_string(),
        is_verified_crypto: false,
        subject: None,
        expires_at: None,
        features: None,
    }
}

/// Resolves the effective license tier from tool arguments or environment variables.
pub fn get_license_tier(args: &Value) -> LicenseTier {
    resolve_license(args).tier
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_compact::Seed;

    const TEST_SEED_BYTES: [u8; 32] = *b"AETRE_TEST_ED25519_SEED_HARNESS_";

    #[test]
    fn test_ed25519_cryptographic_verification() {
        let key_pair = KeyPair::from_seed(Seed::new(TEST_SEED_BYTES));
        let pub_key_bytes = key_pair.pk.as_ref();

        let claims = LicenseClaims {
            sub: "org_sequoia_capital".to_string(),
            tier: LicenseTier::Enterprise,
            features: vec![
                "governor".to_string(),
                "pareto_voi".to_string(),
                "audit".to_string(),
            ],
            iat: 1771400000,
            exp: Some(2147483647), // Year 2038
            max_monthly_evals: None,
        };

        // 1. Generate live signed token
        let token = generate_signed_license(&key_pair, &claims, true);
        assert!(token.starts_with("aetre_live_"));

        // 2. Verify token
        let verified = parse_and_verify_token(&token, pub_key_bytes);
        assert!(verified.is_ok());
        let verified_claims = verified.unwrap();
        assert_eq!(verified_claims.tier, LicenseTier::Enterprise);
        assert_eq!(verified_claims.sub, "org_sequoia_capital");

        // 3. Test tampering (modify payload character)
        let mut tampered = token.clone();
        tampered.replace_range(15..16, "X");
        let tampered_result = parse_and_verify_token(&tampered, pub_key_bytes);
        assert!(tampered_result.is_err());

        // 4. Test expired token
        let expired_claims = LicenseClaims {
            sub: "expired_corp".to_string(),
            tier: LicenseTier::Pro,
            features: vec![],
            iat: 100000,
            exp: Some(100001), // In the deep past
            max_monthly_evals: None,
        };
        let expired_token = generate_signed_license(&key_pair, &expired_claims, true);
        let expired_result = parse_and_verify_token(&expired_token, pub_key_bytes);
        assert_eq!(expired_result, Err(LicenseError::Expired));
    }

    #[test]
    fn test_resolve_license_sources() {
        let key_pair = KeyPair::from_seed(Seed::new(TEST_SEED_BYTES));
        let claims = LicenseClaims {
            sub: "test_research_lab".to_string(),
            tier: LicenseTier::Pro,
            features: vec![],
            iat: 1771400000,
            exp: Some(2147483647),
            max_monthly_evals: None,
        };
        let valid_token = generate_signed_license(&key_pair, &claims, true);
        let pub_key_bytes: &[u8] = key_pair.pk.as_ref();

        // Tool arguments resolution
        let resolved = resolve_license_with_key(&json!({ "api_key": valid_token }), pub_key_bytes);
        assert_eq!(resolved.tier, LicenseTier::Pro);
        assert!(resolved.is_verified_crypto);
        assert_eq!(resolved.subject.as_deref(), Some("test_research_lab"));

        // Demo alias resolution
        let resolved_demo = resolve_license(&json!({ "api_key": "aetre_ent_demo_key" }));
        assert_eq!(resolved_demo.tier, LicenseTier::Enterprise);
        assert!(!resolved_demo.is_verified_crypto);

        // Default Community
        let resolved_default = resolve_license(&json!({}));
        assert_eq!(resolved_default.tier, LicenseTier::Community);
    }

    #[test]
    fn test_root_public_key_integrity() {
        assert_eq!(ROOT_PUBLIC_KEY_BYTES.len(), 32);
        assert!(PublicKey::from_slice(&ROOT_PUBLIC_KEY_BYTES).is_ok());
    }
}
