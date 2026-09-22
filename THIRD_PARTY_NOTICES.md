# Third-party notices

The project's own source code and documentation are licensed under the [GNU AGPL-3.0-or-later](LICENSE). This file records material that is not relicensed by that project licence

## Python dependencies

The Python dependencies are declared in [`requirements.txt`](requirements.txt). They remain under their own licences. The main direct dependencies use permissive or weak-copyleft terms, including BSD, MIT, Apache-2.0, MPL-2.0, and the Python Software Foundation licence. Some distributions also bundle separately licensed native libraries

A deployment must preserve the licence and attribution notices supplied by the exact dependency versions it installs. A lock file, wheel directory, or software bill of materials is the authoritative version-level inventory for a particular build

Notable examples include:

- NumPy and SciPy: BSD-family licences, with native numerical libraries carrying their own notices
- pandas, scikit-learn, networkx, FastAPI, Streamlit, and related Python packages: check the installed distribution metadata for the exact licence
- Requests and OpenTelemetry packages: Apache-2.0 terms or Apache-2.0-compatible project terms
- Redis client, Huey, Plotly, Beautiful Soup, pytest, and tooling: each retains its upstream licence

Do not replace these upstream terms with the AGPL when redistributing an environment or container

## Rust dependencies

The Rust extension is declared in [`src/tournament/tcg_engine/Cargo.toml`](src/tournament/tcg_engine/Cargo.toml). PyO3, rand, and Rayon retain their upstream licences. Cargo resolves their transitive dependency tree when the extension is built

## Contributor Covenant

[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) is adapted from Contributor Covenant 3.0 and remains under the attribution and licence terms stated in that file. It is a community policy document, not project source code

## Limitless data and services

The scraper and ingestion client use data and endpoints provided by Limitless. Limitless data is external to this repository and is not relicensed under the project AGPL. Follow the source site's terms, attribution requirements, rate limits, and applicable law when fetching or redistributing it

Stored snapshots, matchup matrices, decklists, and derived reports can contain database rights or other rights held by their sources or contributors. Keep the source URL, retrieval date, format, and any applicable terms with a published data report

## Docker images and system software

The Compose file pulls Redis and Jaeger images from their upstream registries. Those images and the software they contain have separate licences and notices. Check each image's documentation and digest when distributing a built deployment

## Notice maintenance

Before adding a dependency, data source, Docker image, or copied asset:

1. Record its upstream project and exact version or retrieval date
2. Record its licence and attribution requirements
3. Check compatibility with the project's distribution model
4. Preserve required notices in this file or in the distributed artefact
