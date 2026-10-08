use std::{
    cmp::Ordering,
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(
    name = "qvec",
    version,
    about = "Build and search a SQLite semantic index"
)]
struct Cli {
    /// SQLite database file
    #[arg(short, long, global = true, default_value = "qvec.sqlite")]
    file: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create a new index using a FastEmbed model
    Init {
        /// FastEmbed model name, for example BAAI/bge-small-en-v1.5
        #[arg(long, default_value = "BAAI/bge-small-en-v1.5")]
        model: String,
    },
    /// Add or replace a document, reading its text from stdin
    Insert {
        /// Document key
        key: String,
    },
    /// Add or replace text files, recursively processing directories
    InsertFiles {
        /// One or more text files or directories
        #[arg(required = true)]
        paths: Vec<PathBuf>,
    },
    /// Search for the nearest documents
    Search {
        /// Query text
        query: String,
        /// Maximum number of results
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        /// Print search results as a JSON array
        #[arg(long)]
        json: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init { model } => initialize(&cli.file, &model),
        Command::Insert { key } => {
            let mut text = String::new();
            io::stdin()
                .read_to_string(&mut text)
                .context("reading document from stdin")?;
            insert_texts(&cli.file, vec![(key, text)])
        }
        Command::InsertFiles { paths } => insert_files(&cli.file, &paths),
        Command::Search { query, limit, json } => search(&cli.file, &query, limit, json),
    }
}

fn initialize(db_path: &Path, model_name: &str) -> Result<()> {
    let model = parse_model(model_name)?;
    let db = Connection::open(db_path)
        .with_context(|| format!("opening SQLite database {}", db_path.display()))?;
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS qvec_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            key TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        );",
    )?;
    let existing: Option<String> = db
        .query_row("SELECT model FROM qvec_meta WHERE id = 1", [], |row| {
            row.get(0)
        })
        .optional()?;
    if let Some(existing) = existing {
        if existing != model.to_string() {
            bail!(
                "index uses model {existing}, but {model_name} was requested; create a new index to change models"
            );
        }
    } else {
        db.execute(
            "INSERT INTO qvec_meta (id, model) VALUES (1, ?1)",
            [model.to_string()],
        )?;
    }
    eprintln!("Loading model {}...", model);
    drop(load_embedder(model.clone())?);
    println!("Initialized {} with {}", db_path.display(), model);
    Ok(())
}

fn parse_model(model_name: &str) -> Result<EmbeddingModel> {
    if let Ok(model) = EmbeddingModel::from_str(model_name) {
        return Ok(model);
    }

    // BGE ONNX conversions use Xenova IDs in FastEmbed, while users know the original BAAI IDs.
    let fastembed_id = match model_name.to_ascii_lowercase().as_str() {
        "baai/bge-small-en-v1.5" => "Xenova/bge-small-en-v1.5",
        "baai/bge-base-en-v1.5" => "Xenova/bge-base-en-v1.5",
        "baai/bge-large-en-v1.5" => "Xenova/bge-large-en-v1.5",
        "baai/bge-small-zh-v1.5" => "Xenova/bge-small-zh-v1.5",
        "baai/bge-large-zh-v1.5" => "Xenova/bge-large-zh-v1.5",
        _ => model_name,
    };
    TextEmbedding::list_supported_models()
        .into_iter()
        .find(|info| info.model_code.eq_ignore_ascii_case(fastembed_id))
        .map(|info| info.model)
        .ok_or_else(|| anyhow::anyhow!("unsupported FastEmbed model: {model_name}"))
}

fn open_index(db_path: &Path) -> Result<(Connection, EmbeddingModel)> {
    let db = Connection::open(db_path)
        .with_context(|| format!("opening SQLite database {}", db_path.display()))?;
    let model_name: Option<String> = db
        .query_row("SELECT model FROM qvec_meta WHERE id = 1", [], |row| {
            row.get(0)
        })
        .optional()
        .context("not a qvec index; run `qvec init` first")?;
    let model_name = model_name.context("not a qvec index; run `qvec init` first")?;
    let model = parse_model(&model_name)
        .map_err(|_| anyhow::anyhow!("index contains unsupported model: {model_name}"))?;
    Ok((db, model))
}

fn load_embedder(model: EmbeddingModel) -> Result<TextEmbedding> {
    let cache_dir = qvec_cache_dir()?;
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("creating model cache {}", cache_dir.display()))?;
    TextEmbedding::try_new(TextInitOptions::new(model).with_cache_dir(cache_dir))
        .context("loading embedding model")
}

fn qvec_cache_dir() -> Result<PathBuf> {
    let xdg_cache_home = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from);
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    cache_dir_from(xdg_cache_home.as_deref(), home.as_deref())
}

fn cache_dir_from(xdg_cache_home: Option<&Path>, home: Option<&Path>) -> Result<PathBuf> {
    let base = match xdg_cache_home.filter(|path| path.is_absolute()) {
        Some(path) => path.to_path_buf(),
        None => home
            .filter(|path| path.is_absolute())
            .context("XDG_CACHE_HOME or an absolute HOME directory is required")?
            .join(".cache"),
    };
    Ok(base.join("qvec"))
}

fn insert_files(db_path: &Path, paths: &[PathBuf]) -> Result<()> {
    let cwd = std::env::current_dir().context("getting current directory")?;
    let mut docs = Vec::new();
    for path in paths {
        let absolute = if path.is_absolute() {
            path.clone()
        } else {
            cwd.join(path)
        };
        let metadata =
            fs::metadata(&absolute).with_context(|| format!("reading {}", path.display()))?;
        if metadata.is_dir() {
            for entry in WalkDir::new(&absolute).sort_by_file_name() {
                let entry = entry.with_context(|| format!("walking {}", path.display()))?;
                if entry.file_type().is_file() {
                    let key = file_key(entry.path(), &cwd)?;
                    docs.push(read_file(entry.path(), key)?);
                }
            }
        } else if metadata.is_file() {
            let key = file_key(&absolute, &cwd)?;
            docs.push(read_file(&absolute, key)?);
        } else {
            bail!("{} is not a regular file or directory", path.display());
        }
    }
    if docs.is_empty() {
        bail!("no files found to index");
    }
    insert_texts(db_path, docs)
}

fn file_key(path: &Path, cwd: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("resolving {}", path.display()))?;
    let relative = canonical.strip_prefix(cwd).unwrap_or(&canonical);
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::ParentDir => parts.push("..".to_owned()),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    if parts.is_empty() {
        bail!("could not create a document key for {}", path.display());
    }
    Ok(parts.join("/"))
}

fn read_file(path: &Path, key: String) -> Result<(String, String)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("{} is not a readable UTF-8 text file", path.display()))?;
    Ok((key, text))
}

fn insert_texts(db_path: &Path, docs: Vec<(String, String)>) -> Result<()> {
    let (mut db, model) = open_index(db_path)?;
    if docs.is_empty() {
        bail!("no documents to insert");
    }
    let mut embedder = load_embedder(model)?;
    let texts: Vec<&str> = docs.iter().map(|(_, text)| text.as_str()).collect();
    let embeddings = embedder.embed(texts, None).context("embedding documents")?;
    if embeddings.len() != docs.len() {
        bail!("embedding model returned an unexpected number of vectors");
    }
    let tx = db.transaction()?;
    write_documents(&tx, &docs, &embeddings)?;
    tx.commit()?;
    println!("Inserted {} document(s)", docs.len());
    Ok(())
}

fn write_documents(
    tx: &Transaction<'_>,
    docs: &[(String, String)],
    embeddings: &[Vec<f32>],
) -> Result<()> {
    let mut statement = tx.prepare(
        "INSERT INTO documents (key, text, embedding) VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET text = excluded.text, embedding = excluded.embedding",
    )?;
    for ((key, text), embedding) in docs.iter().zip(embeddings) {
        statement.execute(params![key, text, serde_json::to_string(embedding)?])?;
    }
    Ok(())
}

fn search(db_path: &Path, query: &str, limit: usize, json: bool) -> Result<()> {
    let (db, model) = open_index(db_path)?;
    let mut embedder = load_embedder(model)?;
    let query_embedding = embedder
        .embed(vec![query], None)
        .context("embedding query")?
        .into_iter()
        .next()
        .context("embedding model returned no query vector")?;

    let mut statement = db.prepare("SELECT key, embedding FROM documents")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut results = Vec::new();
    for row in rows {
        let (key, vector_json) = row?;
        let vector: Vec<f32> = serde_json::from_str(&vector_json)
            .with_context(|| format!("invalid stored embedding for key {key:?}"))?;
        if vector.len() != query_embedding.len() {
            bail!("stored embedding for key {key:?} has a different model dimension");
        }
        results.push((cosine_distance(&query_embedding, &vector), key));
    }
    results.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));

    if json {
        let output: Vec<SearchResult> = results
            .into_iter()
            .take(limit)
            .map(|(distance, key)| SearchResult { distance, key })
            .collect();
        let stdout = io::stdout();
        let mut output_stream = stdout.lock();
        serde_json::to_writer(&mut output_stream, &output)?;
        writeln!(output_stream)?;
    } else {
        println!("dist key");
        for (distance, key) in results.into_iter().take(limit) {
            println!("{distance:.2} {key}");
        }
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct SearchResult {
    distance: f32,
    key: String,
}

fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        1.0
    } else {
        1.0 - dot / (left_norm * right_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::{cache_dir_from, cosine_distance, file_key, parse_model};
    use std::{fs, path::Path};

    #[test]
    fn cache_directory_uses_xdg_or_home_fallback() {
        let xdg_cache = std::env::temp_dir().join("xdg-cache");
        let home = std::env::temp_dir().join("qvec-home");
        assert_eq!(
            cache_dir_from(Some(&xdg_cache), Some(&home)).unwrap(),
            xdg_cache.join("qvec")
        );
        assert_eq!(
            cache_dir_from(Some(Path::new("relative")), Some(&home)).unwrap(),
            home.join(".cache/qvec")
        );
    }

    #[test]
    fn serializes_search_results_as_json() {
        let result = super::SearchResult {
            distance: 0.25,
            key: "hello".to_owned(),
        };
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            serde_json::json!({ "distance": 0.25, "key": "hello" })
        );
    }

    #[test]
    fn accepts_the_original_bge_model_id() {
        assert_eq!(
            parse_model("BAAI/bge-small-en-v1.5").unwrap(),
            fastembed::EmbeddingModel::BGESmallENV15
        );
    }

    #[test]
    fn cosine_distance_orders_identical_and_opposite_vectors() {
        assert!(cosine_distance(&[1.0, 0.0], &[1.0, 0.0]).abs() < 1e-6);
        assert!((cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn file_keys_use_forward_slashes_and_relative_paths() {
        let cwd = std::env::temp_dir().join(format!("qvec-test-{}", std::process::id()));
        let file = cwd.join("dir").join("quux.txt");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "text").unwrap();
        assert_eq!(file_key(&file, &cwd).unwrap(), "dir/quux.txt");
        fs::remove_dir_all(cwd).unwrap();
    }
}
