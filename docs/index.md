# Lightning Core Docs Index

Use one of the paths below depending on your goal.

## Quick Links by Task

| Task Goal | Go To | Why |
| --- | --- | --- |
| Install / First Run | [Quickstart: Install](quickstart.md#1-install) | Set up, build, import, and run your first checks quickly. |
| Performance / Tuning | [Advanced: Benchmark Suite](advanced.md#benchmark-suite) | Benchmark, tune runtime/session behavior, and inspect profiles. |
| Capability / Environment Contracts | [Capability Matrix](capability_matrix.md) | Check runtime backend capability surfaces and validated test environments. |
| API Surface Navigation | [API Index](api_index.md) | Jump directly to Python/C++/C API entry points and source links. |
| Contribute / Code Changes | [Contributor: Validation Checklist](contributor.md#5-validation-checklist-before-commit) | Follow contribution workflow, naming rules, and validation steps. |
| Release / Deployment Ops | [Advanced: Python Packaging and Release](advanced.md#python-packaging-and-release) | Use release workflow and repository rename operation guidance. |

## Beginner Path

Start here if you are using the project for the first time.

- [docs/quickstart.md](quickstart.md)

Covers:

- install/import check
- first build
- first C API run
- first test run
- minimal C++ and Python examples

## Advanced Path

Use this when you need tuning, benchmarking, and operations details.

- [docs/advanced.md](advanced.md)

Covers:

- benchmark sweeps and generated artifacts
- resident policy/session tuning
- runtime profile env controls
- model-wrapper caveats
- release workflow and rename operations

## API Navigation Path

Use this when you need to find exact API entry points quickly.

- [docs/api_index.md](api_index.md)

Covers:

- Python API surfaces (`lightning_core`, `lightning_core.api`)
- public C++ headers
- public C API header and implementation mapping

## Contributor Path

Use this when you are changing code and preparing commits.

- [docs/contributor.md](contributor.md)

Covers:

- development setup
- codebase structure and naming policy
- compatibility/shim expectations
- python binding module policy
- validation checklist before commit
