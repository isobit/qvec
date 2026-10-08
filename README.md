# qvec

`qvec` builds a local semantic search index in a SQLite database using FastEmbed.

```sh
qvec -f index.sqlite init --model "BAAI/bge-small-en-v1.5"
qvec -f index.sqlite insert hello <<< "Hello, world!"
qvec -f index.sqlite insert-files foo.txt
qvec -f index.sqlite insert-files dir/
qvec -f index.sqlite search "greeting"
qvec -f index.sqlite search --json "greeting"
```

`insert` reads the document body from standard input. `insert-files` accepts
text files and recursively indexes files in directories, using paths relative
to the current working directory as keys. Inserting a key again replaces its
text and embedding.

Search output is ordered by ascending cosine distance (`1 - cosine similarity`)
and prints a `dist key` header followed by matching rows. Add `--json` to
return a JSON array of `{ "distance": number, "key": string }` objects. The model
is downloaded by FastEmbed on first use and reused from its model cache under
`$XDG_CACHE_HOME/qvec` (`$HOME/.cache/qvec` when `XDG_CACHE_HOME` is unset or relative).
FastEmbed's `HF_HOME` environment variable takes precedence if set.
